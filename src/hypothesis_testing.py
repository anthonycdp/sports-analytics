"""
Statistical Hypothesis Testing Module for Sports Analytics

This module provides reusable functions for conducting statistical
hypothesis tests commonly used in sports analytics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    effect_size: Optional[float] = None
    interpretation: str = ""

    def __str__(self) -> str:
        result = "REJECT" if self.reject_null else "FAIL TO REJECT"
        return (f"{self.test_name}\n"
                f"  Statistic: {self.statistic:.4f}\n"
                f"  p-value: {self.p_value:.6f}\n"
                f"  Result: {result} H0 (α={self.alpha})\n"
                f"  {self.interpretation}")


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two independent groups.

    Parameters
    ----------
    group1, group2 : array-like
        The two groups to compare

    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Parameters
    ----------
    d : float
        Cohen's d value

    Returns
    -------
    str
        Interpretation string
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "Negligible effect"
    elif d_abs < 0.5:
        return "Small effect"
    elif d_abs < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


def independent_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
    alternative: str = 'two-sided'
) -> HypothesisTestResult:
    """
    Perform independent samples t-test.

    Parameters
    ----------
    group1, group2 : array-like
        The two groups to compare
    alpha : float
        Significance level (default: 0.05)
    equal_var : bool
        Assume equal variances (default: False, uses Welch's t-test)
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    HypothesisTestResult
        Test result object
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var,
                                       alternative=alternative)

    d = cohens_d(group1, group2)
    reject = p_value < alpha

    mean_diff = np.mean(group1) - np.mean(group2)

    result = HypothesisTestResult(
        test_name="Independent Samples t-test" + (" (Welch's)" if not equal_var else ""),
        statistic=t_stat,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        effect_size=d,
        interpretation=f"Mean difference: {mean_diff:.3f}, Cohen's d: {d:.3f} ({interpret_cohens_d(d)})"
    )

    return result


def paired_t_test(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> HypothesisTestResult:
    """
    Perform paired samples t-test.

    Parameters
    ----------
    before, after : array-like
        Paired observations
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    HypothesisTestResult
        Test result object
    """
    t_stat, p_value = stats.ttest_rel(before, after, alternative=alternative)

    reject = p_value < alpha
    mean_diff = np.mean(after) - np.mean(before)

    # Cohen's d for paired data
    diff = after - before
    d = np.mean(diff) / np.std(diff, ddof=1)

    result = HypothesisTestResult(
        test_name="Paired Samples t-test",
        statistic=t_stat,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        effect_size=d,
        interpretation=f"Mean change: {mean_diff:.3f}, Cohen's d: {d:.3f} ({interpret_cohens_d(d)})"
    )

    return result


def one_way_anova(
    groups: List[np.ndarray],
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Perform one-way ANOVA.

    Parameters
    ----------
    groups : list of arrays
        Groups to compare
    alpha : float
        Significance level

    Returns
    -------
    HypothesisTestResult
        Test result object
    """
    f_stat, p_value = stats.f_oneway(*groups)

    reject = p_value < alpha

    # Calculate eta-squared (effect size)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total

    result = HypothesisTestResult(
        test_name="One-Way ANOVA",
        statistic=f_stat,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        effect_size=eta_squared,
        interpretation=f"η² (eta-squared): {eta_squared:.3f}"
    )

    return result


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Perform chi-square goodness of fit or test of independence.

    Parameters
    ----------
    observed : array-like
        Observed frequencies
    expected : array-like, optional
        Expected frequencies (uniform if None)
    alpha : float
        Significance level

    Returns
    -------
    HypothesisTestResult
        Test result object
    """
    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)

    reject = p_value < alpha

    # Cramer's V effect size
    n = np.sum(observed)
    min_dim = min(observed.shape) - 1 if len(observed.shape) > 1 else 1
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0

    result = HypothesisTestResult(
        test_name="Chi-Square Test",
        statistic=chi2_stat,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        effect_size=cramers_v,
        interpretation=f"Cramér's V: {cramers_v:.3f}"
    )

    return result


def pearson_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Test significance of Pearson correlation.

    Parameters
    ----------
    x, y : array-like
        Variables to correlate
    alpha : float
        Significance level

    Returns
    -------
    HypothesisTestResult
        Test result object
    """
    r, p_value = stats.pearsonr(x, y)

    reject = p_value < alpha

    # Interpret correlation strength
    r_abs = abs(r)
    if r_abs < 0.1:
        strength = "negligible"
    elif r_abs < 0.3:
        strength = "weak"
    elif r_abs < 0.5:
        strength = "moderate"
    elif r_abs < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if r > 0 else "negative"

    result = HypothesisTestResult(
        test_name="Pearson Correlation Test",
        statistic=r,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        effect_size=r,
        interpretation=f"r = {r:.3f} ({strength} {direction} correlation)"
    )

    return result


def mann_whitney_u_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> HypothesisTestResult:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Parameters
    ----------
    group1, group2 : array-like
        Groups to compare
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    HypothesisTestResult
        Test result object
    """
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

    reject = p_value < alpha

    # Effect size (rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    r_effect = 1 - (2 * u_stat) / (n1 * n2)

    result = HypothesisTestResult(
        test_name="Mann-Whitney U Test",
        statistic=u_stat,
        p_value=p_value,
        alpha=alpha,
        reject_null=reject,
        effect_size=r_effect,
        interpretation=f"Effect size r: {r_effect:.3f}"
    )

    return result


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : list
        List of p-values from multiple tests
    alpha : float
        Family-wise error rate

    Returns
    -------
    tuple
        (List of rejection decisions, adjusted alpha)
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests

    rejections = [p < adjusted_alpha for p in p_values]

    return rejections, adjusted_alpha


def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive comparison of groups including tests and effect sizes.

    Parameters
    ----------
    df : DataFrame
        Data containing groups and values
    group_col : str
        Column name for group labels
    value_col : str
        Column name for values to compare
    alpha : float
        Significance level

    Returns
    -------
    dict
        Dictionary with descriptive stats and test results
    """
    groups = df[group_col].unique()

    # Descriptive statistics
    desc_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'median'])

    results = {
        'descriptive_stats': desc_stats,
        'tests': {}
    }

    if len(groups) == 2:
        # Two groups - t-test
        g1_data = df[df[group_col] == groups[0]][value_col].values
        g2_data = df[df[group_col] == groups[1]][value_col].values

        results['tests']['t_test'] = independent_t_test(g1_data, g2_data, alpha)
        results['tests']['mann_whitney'] = mann_whitney_u_test(g1_data, g2_data, alpha)

    elif len(groups) > 2:
        # Multiple groups - ANOVA
        group_data = [df[df[group_col] == g][value_col].values for g in groups]
        results['tests']['anova'] = one_way_anova(group_data, alpha)

    return results


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)

    # Example: Compare two groups
    group_a = np.random.normal(100, 15, 50)
    group_b = np.random.normal(108, 15, 50)

    print("=" * 60)
    print("HYPOTHESIS TESTING MODULE DEMO")
    print("=" * 60)

    # Independent t-test
    result = independent_t_test(group_a, group_b)
    print(f"\n{result}")

    # Pearson correlation
    x = np.random.normal(0, 1, 100)
    y = x * 0.8 + np.random.normal(0, 0.5, 100)

    corr_result = pearson_correlation_test(x, y)
    print(f"\n{corr_result}")

    # ANOVA
    groups = [
        np.random.normal(50, 10, 30),
        np.random.normal(55, 10, 30),
        np.random.normal(60, 10, 30)
    ]

    anova_result = one_way_anova(groups)
    print(f"\n{anova_result}")
