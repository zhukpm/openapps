import asyncio
import copy
import logging
import time
import pandas as pd
from jinja2 import Environment, FileSystemLoader, meta
from openai.types import CompletionUsage
from pydantic import BaseModel
from typing import Literal, Sequence, Any, Self, overload
from openai import AsyncClient
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from pathlib import Path
import hashlib
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids
import warnings
from dppy.finite_dpps import FiniteDPP
from few_shot_optimization_with_ga.importance import get_importance_scores, Config


warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='sklearn.utils'
)

# Setting up loggers

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Monitoring pricing

class ModelPricing(BaseModel):
    prompt: float
    completion: float
    cached_prompt: float


COMPLETION_PRICE = {
    'gpt-5-nano': ModelPricing(
        prompt=0.05,
        cached_prompt=0.01,
        completion=0.4,
    )
}


class ModelUsage(BaseModel):
    prompt: int = 0
    cached_prompt: int = 0
    completion: int = 0

    def __add__(self, other: 'ModelUsage') -> 'ModelUsage':
        return ModelUsage(
            prompt=self.prompt + other.prompt,
            cached_prompt=self.cached_prompt + other.cached_prompt,
            completion=self.completion + other.completion,
        )


class InferenceSession:
    def __init__(self):
        self.records: list[ModelUsage] = []
        self.total_usage: ModelUsage = ModelUsage()

    def record(self, usage: CompletionUsage) -> None:
        model_usage = ModelUsage(
            prompt=usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens,
            cached_prompt=usage.prompt_tokens_details.cached_tokens,
            completion=usage.completion_tokens
        )
        self.total_usage += model_usage
        self.records.append(model_usage)



class PriceCalculator:
    def __init__(self, sessions: Sequence[InferenceSession], pricing: ModelPricing):
        self.sessions: Sequence[InferenceSession] = sessions
        self.pricing: ModelPricing = pricing

    def _session_price(self, session: InferenceSession) -> float:
        return (
                session.total_usage.prompt * self.pricing.prompt
                + session.total_usage.cached_prompt * self.pricing.cached_prompt
                + session.total_usage.completion * self.pricing.completion
        ) / 1e6

    def __getitem__(self, item: slice) -> float:
        return sum(map(self._session_price, self.sessions[item]))

    def total(self) -> float:
        return self[:]


class TokensWatcher:
    def __init__(self):
        self.sessions: list[InferenceSession] = []

    def new_session(self):
        self.sessions.append(InferenceSession())

    def record(self, usage: CompletionUsage) -> None:
        if not self.sessions:
            raise RuntimeError('No sessions recorded')
        self.sessions[-1].record(usage)

    @property
    def price_gpt5nano(self) -> PriceCalculator:
        return PriceCalculator(self.sessions, COMPLETION_PRICE['gpt-5-nano'])


# Classification response model

ScientificAreaT = Literal[
    'network security', 'parallel computing', 'computer vision',
    'computer programming', 'software engineering', 'electricity',
    'pid controller', 'digital control', 'operational amplifier',
    'system identification', 'attention', 'child abuse',
    'social cognition', 'gender roles', 'nonverbal communication',
    'machine design', 'hydraulics', 'internal combustion engine',
    'fluid mechanics', 'computer-aided design', 'water pollution',
    'rainwater harvesting', 'geotextile', 'green building',
    'construction management', 'fungal infection', 'menopause',
    "alzheimer's disease", 'sports injuries', 'stress management',
    'polymerase chain reaction', 'molecular biology',
    'northern blotting', 'immunology', 'human metabolism'
]


class Response(BaseModel, extra='forbid'):
    scientific_area: ScientificAreaT


# Completion functions

async def complete(
        model: AsyncClient,
        messages: list[dict[str, str]],
        tokens_watcher: TokensWatcher,
) -> ScientificAreaT | Literal['OTHER']:
    try:
        res = await model.chat.completions.parse(
            messages=messages,
            model='gpt-5-nano',
            max_completion_tokens=8192,
            n=1,
            seed=42,
            reasoning_effort='low',
            response_format=Response
        )
        tokens_watcher.record(res.usage)
        return res.choices[0].message.parsed.scientific_area
    except Exception:
        logger.exception('An exception occurred during a request.')
    return 'OTHER'


async def classify_single(
        messages: list[dict[str, str]],
        item: str,
        model: AsyncClient,
        tokens_watcher: TokensWatcher,
        user_turn_template: str = '{item}'
) -> ScientificAreaT | Literal['OTHER']:
    prompt = messages + [{'role': 'user', 'content': user_turn_template.format(item=item)}]
    return await complete(model, prompt, tokens_watcher)


async def classify_many(
        messages: list[dict[str, str]],
        items: Sequence[str],
        model: AsyncClient,
        tokens_watcher: TokensWatcher,
        user_turn_template: str = '{item}'
) -> list[ScientificAreaT | Literal['OTHER']]:
    sem = asyncio.Semaphore(10)
    pbar = tqdm(total=len(items), desc='Classifying items', unit='item')
    async def coro(item: str):
        async with sem:
            res = await classify_single(
                messages=messages,
                item=item,
                model=model,
                tokens_watcher=tokens_watcher,
                user_turn_template=user_turn_template
            )
            pbar.update(1)
            return res
    return list(await asyncio.gather(*(coro(item) for item in items)))


class Embeddings:
    def __init__(self, model: AsyncClient):
        self.model: AsyncClient = model
        self._cache: dict[str, list[float]] = {}
        self._cache_folder: Path = Path(__file__).parent / 'embeddings-cache'
        self.load_cache()

    @overload
    async def get(self, input: str) -> list[float]: ...

    @overload
    async def get(self, input: list[str]) -> list[list[float]]: ...

    async def get(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, str):
            return await self._get_single(input)
        sem = asyncio.Semaphore(50)
        async def coro(inp: str) -> list[float]:
            async with sem:
                return await self._get_single(inp)
        return list(await asyncio.gather(*(coro(inp) for inp in input)))

    async def _get_single(self, input: str) -> list[float]:
        if input not in self._cache:
            res = await self.model.embeddings.create(
                input=input,
                model='text-embedding-3-large'
            )
            self._cache[input] = res.data[0].embedding
            self.save_cache()
        return self._cache[input]

    def save_cache(self) -> None:
        self._cache_folder.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache.keys())
        values = np.array([self._cache[k] for k in keys], dtype=float)

        with (self._cache_folder / 'keys.csv').open('wt') as f:
            writer = csv.writer(f)
            for key in keys:
                writer.writerow([key])

        np.save(self._cache_folder / 'values.npy', values)

    def load_cache(self) -> None:
        keys = self._cache_folder / 'keys.csv'
        values = self._cache_folder / 'values.npy'
        if not keys.exists() or not values.exists():
            return

        with (self._cache_folder / 'keys.csv').open('rt') as f:
            reader = csv.reader(f)
            keys = [row[0] for row in reader]

        values = np.load(self._cache_folder / 'values.npy')

        self._cache = {k: list(v) for k, v in zip(keys, values)}


# metrics

def compute_metrics(
        y_true: Sequence[ScientificAreaT],
        y_pred: Sequence[ScientificAreaT | Literal['OTHER']],
) -> tuple[str, dict[str, float]]:
    string_report = classification_report(y_true, y_pred, output_dict=False, digits=3, zero_division=0)
    dict_report = classification_report(y_true, y_pred, output_dict=True, digits=3, zero_division=0)
    return string_report, dict_report


# helpers

class Timer:
    def __init__(self):
        self.start: float | None = None
        self.end: float | None = None
        self.duration: float | None = None

    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.duration = self.end - self.start


def read_prompt_template(
        file: str,
        template_values: dict[str, Any] | None = None,
):
    template_values = template_values or {}
    template_path = Path(__file__).parent / 'prompts' / file

    if not template_path.exists() or not template_path.is_file():
        raise FileNotFoundError(f'Template file not found: {template_path}')

    # Setup Jinja environment
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
    )

    # Read template content directly
    template_source = template_path.read_text()

    # Parse template to extract variables
    ast = env.parse(template_source)
    required_variables: set[str] = meta.find_undeclared_variables(ast)

    # Check for missing variables
    if missing_vars := required_variables - set(template_values.keys()):
        raise ValueError(f'Missing values for placeholders: {missing_vars} in {template_path}')

    # Check for extra variables
    if extra_vars := set(template_values.keys()) - required_variables:
        raise ValueError(f'Extra values provided for placeholders: {extra_vars} in {template_path}')

    # Render the template using the source and environment
    template = env.from_string(template_source)
    rendered_text = template.render(**template_values)

    return rendered_text


def save_predictions_for_input(
        name: str,
        inputs: Sequence[str],
        predictions: Sequence[ScientificAreaT | Literal['OTHER']],
) -> None:
    predictions_path = Path(__file__).parent / 'predictions'
    predictions_path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5('\n'.join(inputs).encode('utf-8')).hexdigest()
    with open(predictions_path / f'{name}-{digest}.csv', 'w') as f:
        for prediction in predictions:
            f.write(prediction + '\n')


def load_predictions_for_input(
        name: str,
        inputs: Sequence[str]
) -> list[ScientificAreaT | Literal['OTHER']] | None:
    predictions_path = Path(__file__).parent / 'predictions'
    predictions_path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5('\n'.join(inputs).encode('utf-8')).hexdigest()
    predictions_file = predictions_path / f'{name}-{digest}.csv'
    if not predictions_file.exists():
        return None
    return list(map(str.strip, predictions_file.read_text().splitlines()))


# diversity

def select_k_medoids(
        embeddings: np.ndarray,
        k: int,
        random_seed: int = 42
) -> list[int]:
    kmedoids = KMedoids(
        n_clusters=k,
        metric='cosine',
        method='pam',
        init='k-medoids++',
        max_iter=500,
        random_state=random_seed
    )
    kmedoids.fit(embeddings)
    return kmedoids.medoid_indices_.tolist()

# models

class ZeroShotModel:
    # acc 0.631 / f1-macro 0.632
    def __init__(self):
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]

    async def predict(
            self,
            items: Sequence[str],
            model: AsyncClient,
            tokens_watcher: TokensWatcher,
            user_turn_template: str = '{item}'
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        return await classify_many(
            messages=self.messages,
            items=items,
            model=model,
            tokens_watcher=tokens_watcher,
            user_turn_template=user_turn_template
        )

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        openai_model = AsyncClient()
        tokens_watcher = TokensWatcher()
        name = 'zero-shot'

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    model=openai_model,
                    tokens_watcher=tokens_watcher,
                    user_turn_template='Classify the following set of keywords: {item}'
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])


class RandomFewShotModel:
    # k=70, per_class=False, random_seed=42
    # acc 0.636 / f1-macro 0.624
    # k=2, per_class=True, random_seed=42
    # acc 0.645 / f1-macro 0.637
    def __init__(self, k: int, per_class: bool = False):
        assert k > 0, 'Number of few-shots must be greater than 0'
        self.k = k
        self.per_class = per_class
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]
        self.few_shots: list[tuple[str, ScientificAreaT]] = []
        self.few_shots_hash: str = ''

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
            random_seed: int = 42
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'

        np.random.seed(random_seed)
        if self.per_class:
            class_to_indices: dict[ScientificAreaT, list[int]] = {}
            for idx, label in enumerate(y):
                class_to_indices.setdefault(label, []).append(idx)

            for label, indices in class_to_indices.items():
                if self.k > len(indices):
                    raise ValueError(f'K value ({self.k}) is greater than the number of samples for class "{label}" ({len(indices)})')
                selected_indices = np.random.choice(np.array(indices, dtype=np.int64), self.k, replace=False)
                for inp, lbl in zip(np.array(X)[selected_indices], np.array(y)[selected_indices], strict=True):
                    self.few_shots.append((inp, lbl))
                    self.messages.extend([
                        {'role': 'user', 'content': inp},
                        {'role': 'assistant', 'content': Response(scientific_area=lbl).model_dump_json()},
                    ])
        else:
            indices = np.random.choice(np.arange(len(X), dtype=np.int64), self.k, replace=False)
            for inp, label in zip(np.array(X)[indices], np.array(y)[indices], strict=True):
                self.few_shots.append((inp, label))
                self.messages.extend([
                    {'role': 'user', 'content': inp},
                    {'role': 'assistant', 'content': Response(scientific_area=label).model_dump_json()},
                ])

        dumped_few_shots = '\n'.join(map(lambda pair: f'{pair[0]}->{pair[1]}', self.few_shots))
        self.few_shots_hash = hashlib.md5(dumped_few_shots.encode('utf-8')).hexdigest()

    async def predict(self,
            items: Sequence[str],
            model: AsyncClient,
            tokens_watcher: TokensWatcher,
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        return await classify_many(
            messages=self.messages,
            items=items,
            model=model,
            tokens_watcher=tokens_watcher,
            user_turn_template='{item}'
        )

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        openai_model = AsyncClient()
        tokens_watcher = TokensWatcher()
        name = 'few-shot-' + self.few_shots_hash

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    model=openai_model,
                    tokens_watcher=tokens_watcher,
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])


class KNearestNeighboursModel:
    # k = 20
    # acc 0.707 / f1-macro 0.696
    def __init__(self, k: int):
        assert k > 0, 'Number of nearest neighbors must be greater than 0'
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]
        self.k = k
        self.train_inputs: list[str] = []
        self.train_labels: list[ScientificAreaT] = []
        self.train_input_embeddings: np.ndarray | None = None
        self.train_hash: str = ''
        self.openai_model: AsyncClient = AsyncClient()
        self.embedder = Embeddings(self.openai_model)

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'

        if self.k > len(X):
            raise ValueError(f'K value ({self.k}) is greater than the number of training samples ({len(X)})')

        self.train_inputs = [inp.strip() for inp in X]
        self.train_labels = [label for label in y]
        self.train_input_embeddings = np.array(await self.embedder.get(self.train_inputs))

        dumped_train = '\n'.join(map(lambda pair: f'{pair[0]}->{pair[1]}', zip(X, y, strict=True)))
        self.train_hash = hashlib.md5(dumped_train.encode('utf-8')).hexdigest()

    @staticmethod
    def top_k_indices(arr: np.ndarray, k: int) -> np.ndarray:
        idx_unsorted = np.argpartition(arr, -k)[-k:]
        idx_sorted = idx_unsorted[np.argsort(-arr[idx_unsorted])]
        return idx_sorted

    async def predict_single(
            self,
            item: str,
            tokens_watcher: TokensWatcher,
    ) -> ScientificAreaT | Literal['OTHER']:
        item_emb = np.array(await self.embedder.get(item))
        closest = self.top_k_indices(self.train_input_embeddings @ item_emb, self.k)
        messages = copy.deepcopy(self.messages)
        for i in reversed(closest):
            messages.extend([
                {'role': 'user', 'content': self.train_inputs[i]},
                {'role': 'assistant', 'content': Response(scientific_area=self.train_labels[i]).model_dump_json()},
            ])
        return await classify_single(
            messages=messages,
            item=item.strip(),
            model=self.openai_model,
            tokens_watcher=tokens_watcher,
            user_turn_template='{item}'
        )

    async def predict(self,
            items: Sequence[str],
            tokens_watcher: TokensWatcher,
    ) -> list[ScientificAreaT | Literal['OTHER']]:

        sem = asyncio.Semaphore(30)
        async def coro(inp):
            async with sem:
                return await self.predict_single(inp, tokens_watcher)

        return list(await asyncio.gather(*(coro(inp) for inp in items)))

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        tokens_watcher = TokensWatcher()
        name = f'knn-{self.k}-' + self.train_hash

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    tokens_watcher=tokens_watcher,
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])


class LogisticRegressionModel:
    # n_dim = 100
    # acc 0.734 / f1-macro 0.715
    def __init__(self, n_dim: int) -> None:
        self.n_dim = n_dim
        self.embedder: Embeddings = Embeddings(AsyncClient())
        self.svd: TruncatedSVD | None = None
        self.classifier: LogisticRegression | None = None

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'

        print('Computing embeddings...')
        embeddings = np.array(await self.embedder.get(list(X)))

        print('Performing dimensionality reduction...')
        self.svd = TruncatedSVD(n_components=self.n_dim, n_iter=200, random_state=42)
        X_reduced = self.svd.fit_transform(embeddings)

        print('Training classifier...')
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(X_reduced, y)

        print('Model training completed.')

    async def predict(
            self,
            items: Sequence[str],
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        assert self.embedder is not None, 'Model is not fitted yet'
        assert self.svd is not None, 'Model is not fitted yet'
        assert self.classifier is not None, 'Model is not fitted yet'

        embeddings = np.array(await self.embedder.get(list(items)))
        X_reduced = self.svd.transform(embeddings)
        return self.classifier.predict(X_reduced).tolist()

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        with Timer() as timer:
            predictions = await self.predict(items=inputs)

        metrics = compute_metrics(y_true, predictions)

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])

    async def cross_entropy(self, items: Sequence[str], y_true: Sequence[ScientificAreaT]) -> list[float]:
        """ Returns the cross-entropy loss for each item """
        assert self.embedder is not None, 'Model is not fitted yet'
        assert self.svd is not None, 'Model is not fitted yet'
        assert self.classifier is not None, 'Model is not fitted yet'

        embeddings = np.array(await self.embedder.get(list(items)))
        X_reduced = self.svd.transform(embeddings)
        probas = self.classifier.predict_proba(X_reduced)
        class_indices = {cls: idx for idx, cls in enumerate(self.classifier.classes_)}
        losses = []
        for true_label, proba in zip(y_true, probas):
            true_index = class_indices[true_label]
            loss = -np.log(proba[true_index] + 1e-15)
            losses.append(loss)
        return losses


class HardestFewShotModel:
    """ Uses "hardest" samples from the training set as few-shots based on
    cross-entropy loss (obtained from Logistic Regression).
    If per_class is True, selects k hardest samples per class.
    If per_class is False, selects k hardest samples overall.
    """
    # k=70, per_class=False
    # acc 0.582 / f1-macro 0.589
    # k=2, per_class=True
    # acc 0.602 / f1-macro 0.594
    def __init__(self, k: int = 2, per_class: bool = True):
        self.k = k
        self.per_class = per_class
        self.few_shots: list[tuple[str, ScientificAreaT]] = []
        self.few_shots_hash: str = ''
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]
        self.openai_model: AsyncClient = AsyncClient()

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'

        # Train logistic regression model to get cross-entropy losses
        log_reg_model = LogisticRegressionModel(n_dim=100)
        await log_reg_model.fit(X, y)
        losses = await log_reg_model.cross_entropy(X, y)

        df = pd.DataFrame({
            'input': X,
            'label': y,
            'loss': losses
        })

        if self.per_class:
            hardest_samples = df.groupby('label')[df.columns].apply(
                lambda group: group.nlargest(self.k, 'loss')
            ).reset_index(drop=True)
        else:
            hardest_samples = df.nlargest(self.k, 'loss')

        for _, row in hardest_samples.sort_values('loss').iterrows():
            self.few_shots.append((row['input'], row['label']))
            self.messages.extend([
                {'role': 'user', 'content': row['input']},
                {'role': 'assistant', 'content': Response(scientific_area=row['label']).model_dump_json()},
            ])

        dumped_few_shots = '\n'.join(map(lambda pair: f'{pair[0]}->{pair[1]}', self.few_shots))
        self.few_shots_hash = hashlib.md5(dumped_few_shots.encode('utf-8')).hexdigest()

    async def predict(
            self,
            items: Sequence[str],
            tokens_watcher: TokensWatcher,
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        return await classify_many(
            messages=self.messages,
            items=items,
            model=self.openai_model,
            tokens_watcher=tokens_watcher,
            user_turn_template='{item}'
        )

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        tokens_watcher = TokensWatcher()
        name = 'hardest-few-shot-' + self.few_shots_hash

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    tokens_watcher=tokens_watcher,
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])


class DiverseMedoidsFewShotModel:
    """ Uses diverse samples from the training set as few-shots based on
    K-Medoids selection in the embedding space.
    If per_class is True, selects k diverse samples per class.
    If per_class is False, selects k diverse samples overall.
    """
    # k=70, per_class=False, random_seed=42
    # acc 0.634 / f1-macro 0.626
    # k=2, per_class=True, random_seed=42
    # acc 0.661 / f1-macro 0.658

    def __init__(self, k: int = 2, per_class: bool = True):
        self.k = k
        self.per_class = per_class
        self.few_shots: list[tuple[str, ScientificAreaT]] = []
        self.few_shots_hash: str = ''
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]
        self.openai_model: AsyncClient = AsyncClient()
        self.embedder = Embeddings(self.openai_model)

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
            random_seed: int = 42
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'
        X = list(X)

        if self.per_class:
            for cls in set(y):
                class_samples = [inp for inp, label in zip(X, y, strict=True) if label == cls]
                class_embeddings = np.array(await self.embedder.get(class_samples))
                selected_indices = select_k_medoids(class_embeddings, self.k, random_seed=random_seed)
                for idx in selected_indices:
                    inp = class_samples[idx]
                    self.few_shots.append((inp, cls))
                    self.messages.extend([
                        {'role': 'user', 'content': inp},
                        {'role': 'assistant', 'content': Response(scientific_area=cls).model_dump_json()},
                    ])
        else:
            all_embeddings = np.array(await self.embedder.get(X))
            selected_indices = select_k_medoids(all_embeddings, self.k, random_seed=random_seed)
            for idx in selected_indices:
                inp = X[idx]
                label = y[idx]
                self.few_shots.append((inp, label))
                self.messages.extend([
                    {'role': 'user', 'content': inp},
                    {'role': 'assistant', 'content': Response(scientific_area=label).model_dump_json()},
                ])

        dumped_few_shots = '\n'.join(map(lambda pair: f'{pair[0]}->{pair[1]}', self.few_shots))
        self.few_shots_hash = hashlib.md5(dumped_few_shots.encode('utf-8')).hexdigest()

    async def predict(
            self,
            items: Sequence[str],
            tokens_watcher: TokensWatcher,
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        return await classify_many(
            messages=self.messages,
            items=items,
            model=self.openai_model,
            tokens_watcher=tokens_watcher,
            user_turn_template='{item}'
        )

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        tokens_watcher = TokensWatcher()
        name = 'diverse-medoids-few-shot-' + self.few_shots_hash

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    tokens_watcher=tokens_watcher,
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])


class DiverseDPPFewShotModel:
    """ Uses diverse samples from the training set as few-shots based on
    Determinantal Point Processes (DPP) selection in the embedding space.
    If per_class is True, selects k diverse samples per class.
    If per_class is False, selects k diverse samples overall.
    """
    # k=70, per_class=False, random_seed=42
    # acc 0.657 / f1-macro 0.655
    # k=2, per_class=True, random_seed=42
    # acc 0.621 / f1-macro 0.618

    def __init__(self, k: int = 2, per_class: bool = True):
        self.k = k
        self.per_class = per_class
        self.few_shots: list[tuple[str, ScientificAreaT]] = []
        self.few_shots_hash: str = ''
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]
        self.openai_model: AsyncClient = AsyncClient()
        self.embedder = Embeddings(self.openai_model)

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
            random_seed: int = 42
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'
        X = list(X)

        if self.per_class:
            for cls in set(y):
                class_samples = [inp for inp, label in zip(X, y, strict=True) if label == cls]
                class_embeddings = np.array(await self.embedder.get(class_samples))
                dpp = FiniteDPP('likelihood', L=(class_embeddings @ class_embeddings.T))
                dpp.sample_exact_k_dpp(size=self.k, random_state=random_seed)
                selected_indices = dpp.list_of_samples[0]
                for idx in selected_indices:
                    inp = class_samples[idx]
                    self.few_shots.append((inp, cls))
                    self.messages.extend([
                        {'role': 'user', 'content': inp},
                        {'role': 'assistant', 'content': Response(scientific_area=cls).model_dump_json()},
                    ])
        else:
            all_embeddings = np.array(await self.embedder.get(X))
            dpp = FiniteDPP('likelihood', L=(all_embeddings @ all_embeddings.T))
            dpp.sample_exact_k_dpp(size=self.k, random_state=random_seed)
            selected_indices = dpp.list_of_samples[0]
            for idx in selected_indices:
                inp = X[idx]
                label = y[idx]
                self.few_shots.append((inp, label))
                self.messages.extend([
                    {'role': 'user', 'content': inp},
                    {'role': 'assistant', 'content': Response(scientific_area=label).model_dump_json()},
                ])

        dumped_few_shots = '\n'.join(map(lambda pair: f'{pair[0]}->{pair[1]}', self.few_shots))
        self.few_shots_hash = hashlib.md5(dumped_few_shots.encode('utf-8')).hexdigest()

    async def predict(
            self,
            items: Sequence[str],
            tokens_watcher: TokensWatcher,
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        return await classify_many(
            messages=self.messages,
            items=items,
            model=self.openai_model,
            tokens_watcher=tokens_watcher,
            user_turn_template='{item}'
        )

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        tokens_watcher = TokensWatcher()
        name = 'diverse-dpp-few-shot-' + self.few_shots_hash

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    tokens_watcher=tokens_watcher,
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])


class ImportanceBasedFewShotModel:
    """
    Uses important samples from the training set as few-shots based on
    importance scores computed from the embedding space.
    """
    # k=70, per_class=False, random_seed=42
    # acc 0.663 / f1-macro 0.657
    # k=2, per_class=True, random_seed=42
    # acc 0.644 / f1-macro 0.631
    def __init__(self, config: Config):
        self.config = config
        self.few_shots: list[tuple[str, ScientificAreaT]] = []
        self.few_shots_hash: str = ''
        self.messages: list[dict[str, str]] = [
            {'role': 'system', 'content': read_prompt_template('zero-shot-system.md.jinja')}
        ]
        self.openai_model: AsyncClient = AsyncClient()
        self.embedder = Embeddings(self.openai_model)

    async def fit(
            self,
            X: Sequence[str],
            y: Sequence[ScientificAreaT],
    ) -> None:
        assert len(X) == len(y), 'Inputs and labels must have same length'

        print('Computing embeddings...')
        embeddings = np.array(await self.embedder.get(list(X)))

        print('Computing importance scores...')
        importance_scores = get_importance_scores(embeddings, y, self.config) # N x 3

        importance_weights = np.array(self.config.importance_weights) # shape (3,)
        importance_score = importance_scores @ importance_weights # shape (N,)
        if not self.config.importance_per_class:
            most_important_indices = np.argsort(-importance_score)[:self.config.num_few_shots]
        else:
            most_important_indices = []
            for cls in set(y):
                class_indices = [idx for idx, label in enumerate(y) if label == cls]
                class_importance_scores = importance_score[class_indices]
                selected_indices = np.argsort(-class_importance_scores)[:self.config.num_few_shots]
                most_important_indices.extend(np.array(class_indices)[selected_indices].tolist())
            np.random.RandomState(self.config.random_state).shuffle(most_important_indices)

        print(f'Selected {len(most_important_indices)} indices for few-shots.')
        for idx in most_important_indices:
            inp = X[idx]
            label = y[idx]
            self.few_shots.append((inp, label))
            self.messages.extend([
                {'role': 'user', 'content': inp},
                {'role': 'assistant', 'content': Response(scientific_area=label).model_dump_json()},
            ])

        dumped_few_shots = '\n'.join(map(lambda pair: f'{pair[0]}->{pair[1]}', self.few_shots))
        self.few_shots_hash = hashlib.md5(dumped_few_shots.encode('utf-8')).hexdigest()

    async def predict(
            self,
            items: Sequence[str],
            tokens_watcher: TokensWatcher,
    ) -> list[ScientificAreaT | Literal['OTHER']]:
        return await classify_many(
            messages=self.messages,
            items=items,
            model=self.openai_model,
            tokens_watcher=tokens_watcher,
            user_turn_template='{item}'
        )

    async def score(self, inputs: Sequence[str], y_true: Sequence[ScientificAreaT]) -> None:
        assert len(inputs) == len(y_true), 'Inputs and labels must have same length'

        tokens_watcher = TokensWatcher()
        name = 'importance-based-few-shot-' + self.few_shots_hash

        tokens_watcher.new_session()
        with Timer() as timer:
            predictions = load_predictions_for_input(name, inputs)
            if predictions is None:
                predictions = await self.predict(
                    items=inputs,
                    tokens_watcher=tokens_watcher,
                )
                save_predictions_for_input(name, inputs, predictions)

        metrics = compute_metrics(y_true, predictions)

        price = tokens_watcher.price_gpt5nano.total()

        print(
            f'Prediction took {timer.duration:.1f} seconds. '
            f'Price: {price:.4f} USD. '
            f'F1-macro: {metrics[1]['macro avg']['f1-score']:.3f}'
        )
        print(metrics[0])

# main

async def main():
    dataset_train = pd.read_csv(Path(__file__).parent.parent / 'datasets' / 'fsga_train.csv')
    dataset_val = pd.read_csv(Path(__file__).parent.parent / 'datasets' / 'fsga_val.csv')
    dataset_test = pd.read_csv(Path(__file__).parent.parent / 'datasets' / 'fsga_test.csv')

    # model = RandomFewShotModel(k=2, per_class=True)
    # model = ZeroShotModel()
    # model = KNearestNeighboursModel(k=20)
    # model = LogisticRegressionModel(n_dim=100)
    # model = HardestFewShotModel(k=2, per_class=True)
    # model = DiverseMedoidsFewShotModel(k=2, per_class=True)
    # model = DiverseDPPFewShotModel(k=70, per_class=False)
    model = ImportanceBasedFewShotModel(
        config=Config(
            num_few_shots=70,
            importance_per_class=False,
            importance_weights=(0.4, 0.5, 0.1),
            knn_k=20,
            density_k=7
        )
    )

    await model.fit(
        dataset_train.keywords,
        dataset_train.area,
    )

    await model.score(dataset_test.keywords, dataset_test.area)


if __name__ == '__main__':
    asyncio.run(main())
