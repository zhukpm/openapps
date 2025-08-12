import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Logger:
    def __init__(self):
        self._metrics = {}

    def log(self, metric: str, value: float, iteration: int):
        if metric not in self._metrics:
            self._metrics[metric] = {}
        self._metrics[metric][iteration] = value


def generate_games(means: np.ndarray, scales: np.ndarray, n_games: int) -> np.ndarray:
    winners = []
    losers = []
    # creating an array of players indices
    players = np.arange(means.size)
    for i in range(n_games):
        # taking 2 players at random
        ps = np.random.choice(players, 2, False)
        # generating skills for these players using normal distribution
        skills = np.random.normal(loc=means[ps], scale=scales[ps])
        # winner is the player with higher skill, loser is the one with lower skill
        winners.append(ps[np.argmax(skills)])
        losers.append(ps[np.argmin(skills)])
    return np.vstack([winners, losers]).transpose()


def norm_diff_argument(theta: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    # returns - (mu1 - mu2) / sqrt(sigma1^2 + sigma2^2) for each game in the dataset
    means = theta[:, 0]
    scales = theta[:, 1]
    winners = dataset[:, 0]
    losers = dataset[:, 1]
    return -(means[winners] - means[losers]) / np.sqrt(scales[winners] ** 2 + scales[losers] ** 2)


def Phi(x: np.ndarray) -> np.ndarray:
    # returns the Cumulative Distribution Function of the standard normal distribution at the given point
    return norm.cdf(x, loc=0, scale=1)


def phi(x: np.ndarray) -> np.ndarray:
    # returns the Probability Density Function of the standard normal distribution at the given point
    return norm.pdf(x, loc=0, scale=1)


def log_likelihood(theta: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    x = norm_diff_argument(theta, dataset)
    return np.log(1 - Phi(x))


def grad_log_likelihood(theta: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    players = np.arange(theta.shape[0]).reshape(-1, 1)
    scales = theta[:, 1]
    winners = dataset[:, 0]
    losers = dataset[:, 1]

    x = norm_diff_argument(theta, dataset)

    # common multipliers for both gradients
    phi_by_one_minus_Phi = phi(x) / (1 - Phi(x))

    # -1/0/1 matrix indicating which player is winner/none/loser in each game
    players_x_games = (players == winners).astype(int) - (players == losers).astype(int)

    # calculating gradients for means
    means_grad = players_x_games @ (phi_by_one_minus_Phi / np.sqrt(scales[winners] ** 2 + scales[losers] ** 2))

    # calculating gradients for scales
    scales_grad = np.abs(players_x_games) @ (phi_by_one_minus_Phi * x / (scales[winners] ** 2 + scales[losers] ** 2)) * scales

    # returning gradients as a concatenated array in the form of theta
    return np.vstack([means_grad, scales_grad]).transpose()


def optimize(
        n_players: int,
        dataset: np.ndarray,
        alpha: float = 1e-4,
        epochs: int = 1000,
        logger: Logger | None = None,
) -> np.ndarray:
    # setting theta to an arbitrary value - mean 0 and scale 1 for each player
    theta = np.full((n_players, 2), [0.0, 1.0])

    iteration = 1
    while iteration <= epochs:
        # calculating gradients of log likelihood
        grads = grad_log_likelihood(theta, dataset)

        # updating theta
        theta += alpha * grads

        # calculating likelihood
        likelihood = log_likelihood(theta, dataset)

        # log iteration values
        if logger:
            logger.log('log_likelihood', likelihood.mean(), iteration)
            logger.log('grad_norm', np.linalg.norm(grads), iteration)

        # incrementing iteration
        iteration += 1
    return theta


def plot(
        *args,
        diagram = plt.plot,
        label: str = None,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        **kwargs
):
    plt.figure(figsize=(10, 5))
    diagram(*args, label=label, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(label.replace(' ', '_').lower() + '.png')


def print_metrics(theta: np.ndarray, optimized_theta: np.ndarray):
    print(f'Initial means: {theta[:, 0]}')
    print(f'Optimized means: {optimized_theta[:, 0]}')
    print(f'Initial scales: {theta[:, 1]}')
    print(f'Optimized scales: {optimized_theta[:, 1]}')

    p_diff = []
    for p1 in range(theta.shape[0] - 1):
        for p2 in range(p1 + 1, theta.shape[0]):
            p_real = 1 - Phi(-(theta[p1, 0] - theta[p2, 0]) / np.sqrt(theta[p1, 1] ** 2 + theta[p2, 1] ** 2))
            p_optimized = 1 - Phi(-(optimized_theta[p1, 0] - optimized_theta[p2, 0]) / np.sqrt(optimized_theta[p1, 1] ** 2 + optimized_theta[p2, 1] ** 2))
            p_diff.append(p_real - p_optimized)

    print(f'Mean difference in probabilities: {np.mean(np.abs(p_diff)):.4f}')
    print(f'Median difference in probabilities: {np.median(np.abs(p_diff)):.4f}')

    plot(
        p_diff,
        diagram=plt.hist,
        bins=10,
        alpha=0.7,
        label='Probability Differences',
        xlabel='Probability Difference',
        ylabel='Value',
        title='Distribution of Probability Differences'
    )


if __name__ == '__main__':
    # Example usage
    n_players = 10
    n_games = 300

    # Randomly generating means and scales for players
    means = np.random.uniform(0, 5, n_players)
    scales = np.random.uniform(0.5, 2, n_players)

    # Generating games dataset
    games_dataset = generate_games(means, scales, n_games)

    # Initializing logger
    logger = Logger()

    # Optimizing player skills
    optimized_theta = optimize(n_players, games_dataset, alpha=1e-4, epochs=5000, logger=logger)

    # Printing results
    print_metrics(np.vstack([means, scales]).transpose(), optimized_theta)

    # plotting the results for visualization of logs - saving into a file
    plot(
        list(logger._metrics['log_likelihood'].keys()),
        list(logger._metrics['log_likelihood'].values()),
        label='Log Likelihood',
        xlabel='Iteration',
        ylabel='Value',
        title='Log Likelihood Over Iterations'
    )
    plot(
        list(logger._metrics['grad_norm'].keys()),
        list(logger._metrics['grad_norm'].values()),
        label='Gradient Norm',
        xlabel='Iteration',
        ylabel='Value',
        title='Gradient Norm Over Iterations'
    )
