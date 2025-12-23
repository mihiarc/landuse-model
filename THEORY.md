# Theoretical Foundation: Discrete Choice Land Use Transition Model

This document provides the theoretical foundation for the land use transition model, based on the Random Utility Model (RUM) framework as presented in Train (2009) and applied to land use economics by Lubowski, Plantinga, and Stavins (2006, 2008).

## 1. Random Utility Model Framework

### 1.1 Decision-Maker and Choice Set

The fundamental unit of analysis is a **plot of land** observed at two points in time. The landowner is assumed to be a utility-maximizing agent who chooses the land use that provides the highest expected returns.

**Choice Set (5-Category Model):**
| Code | Use | Description |
|------|-----|-------------|
| 1 | CR | Cropland (combined irrigated + non-irrigated) |
| 2 | PS | Pastureland |
| 3 | RG | Rangeland |
| 4 | FR | Forest |
| 5 | UR | Urban/Developed |

**Key Assumption - Urban Irreversibility:** Once land is converted to urban use, it is assumed to never revert to agricultural or natural uses. This reflects the high cost of de-development and is standard in land use economics. Therefore, no model is estimated for land starting in urban use.

### 1.2 Utility Specification

Following Train (2009, Chapter 2), the utility that landowner *n* derives from land use *j* is decomposed into systematic and random components:

$$U_{nj} = V_{nj} + \varepsilon_{nj}$$

Where:
- $U_{nj}$ = Total utility of alternative *j* for decision-maker *n*
- $V_{nj}$ = Systematic (observable) component of utility
- $\varepsilon_{nj}$ = Random (unobservable) component

The systematic component captures factors the researcher observes:

$$V_{nj} = \beta' X_{nj}$$

Where:
- $X_{nj}$ = Vector of observed attributes (land quality, economic returns)
- $\beta$ = Vector of parameters to be estimated

### 1.3 Choice Probability

The probability that landowner *n* chooses alternative *i* is the probability that the utility of *i* exceeds the utility of all other alternatives:

$$P_{ni} = \Pr(U_{ni} > U_{nj} \text{ for all } j \neq i)$$

$$P_{ni} = \Pr(V_{ni} + \varepsilon_{ni} > V_{nj} + \varepsilon_{nj} \text{ for all } j \neq i)$$

The form of this probability depends on the assumed distribution of the error terms $\varepsilon$.

## 2. Multinomial Logit Model

### 2.1 Distributional Assumption

The multinomial logit (MNL) model assumes that each $\varepsilon_{nj}$ is independently and identically distributed (iid) with a Type I Extreme Value (Gumbel) distribution:

$$f(\varepsilon_{nj}) = e^{-\varepsilon_{nj}} e^{-e^{-\varepsilon_{nj}}}$$

### 2.2 Choice Probability Formula

Under this distributional assumption, the choice probability has the closed-form logit formula (Train 2009, Chapter 3):

$$P_{ni} = \frac{e^{V_{ni}}}{\sum_{j=1}^{J} e^{V_{nj}}}$$

This is the **softmax** function, ensuring:
- $0 < P_{ni} < 1$ for all alternatives
- $\sum_{j=1}^{J} P_{ni} = 1$

### 2.3 Independence of Irrelevant Alternatives (IIA)

A key property of multinomial logit is IIA: the ratio of probabilities for any two alternatives is independent of other alternatives:

$$\frac{P_{ni}}{P_{nk}} = \frac{e^{V_{ni}}}{e^{V_{nk}}} = e^{V_{ni} - V_{nk}}$$

**Implications for Land Use:**
- Adding or removing an alternative (e.g., a new land use category) does not affect the relative odds between existing alternatives
- This may be restrictive if some land uses are closer substitutes (e.g., crop and pasture may be more substitutable than crop and urban)

### 2.4 Maximum Likelihood Estimation

Parameters are estimated by maximum likelihood. Let $y_{ni} = 1$ if person *n* chose alternative *i*, and 0 otherwise. The log-likelihood function is:

$$\mathcal{L}(\beta) = \sum_{n=1}^{N} \sum_{j=1}^{J} y_{nj} \ln P_{nj}$$

$$\mathcal{L}(\beta) = \sum_{n=1}^{N} \sum_{j=1}^{J} y_{nj} \left( V_{nj} - \ln \sum_{k=1}^{J} e^{V_{nk}} \right)$$

## 3. Application to Land Use Transitions

### 3.1 Model Structure

We estimate **separate models for each starting land use**. For land starting in use *s*, we model the transition to end use *e*:

$$P(e | s) = \frac{e^{V_{se}}}{\sum_{k=1}^{J} e^{V_{sk}}}$$

This approach:
- Allows different transition dynamics from different starting uses
- Recognizes that the factors driving crop-to-urban conversion may differ from forest-to-urban conversion
- Follows Lubowski et al. (2006, 2008) methodology

### 3.2 Systematic Utility Specification

The systematic utility of transitioning from start use *s* to end use *e* is:

$$V_{se} = \alpha_e + \beta_e \cdot \text{LCC} + \gamma_e \cdot \text{NR}_e$$

Where:
- $\alpha_e$ = Alternative-specific constant for end use *e*
- $\text{LCC}$ = Land Capability Class (1-8 scale, lower = better agricultural quality)
- $\text{NR}_e$ = Net returns to land use *e* ($/acre)
- $\beta_e, \gamma_e$ = Parameters to estimate

**Note on Identification:** One alternative must be normalized (set as base category) since only utility differences matter in discrete choice. We normalize to the "stay in current use" alternative.

### 3.3 Model Specifications by Starting Use

Based on data availability and economic sign validation:

| Starting Use | Specification | Rationale |
|--------------|---------------|-----------|
| Crop (CR) | LCC + nr_ur | Urban returns drive conversion decisions |
| Pasture (PS) | LCC + nr_ur | Urban returns drive conversion decisions |
| Range (RG) | LCC only | No reliable rent data for rangeland |
| Forest (FR) | LCC only | No reliable rent data for forest |

**Why only urban net returns (nr_ur)?**
- Urban development represents the primary "pull" factor for land conversion
- Agricultural-to-agricultural transitions are driven more by land quality (LCC)
- Including all net returns caused coefficient sign violations in many regions

### 3.4 Expected Coefficient Signs

**Land Capability Class (LCC):**
- For agricultural end uses: $\beta_e < 0$ (lower LCC = better quality = higher probability)
- For urban end use: $\beta_e$ ambiguous (urban development less sensitive to agricultural quality)

**Urban Net Returns (nr_ur):**
- For urban end use: $\gamma_{ur} > 0$ (higher urban returns = higher probability of urban conversion)
- This is the key economic prediction: land converts to uses with higher returns

### 3.5 Regional Stratification

Models are estimated separately by RPA (Resources Planning Act) subregion to capture:
- Regional variation in land markets
- Different agricultural systems and land use patterns
- Varying urbanization pressures

**RPA Subregions:**
| Code | Region |
|------|--------|
| NE | Northeast |
| LS | Lake States |
| CB | Corn Belt |
| NP | Northern Plains |
| AP | Appalachian |
| SE | Southeast |
| DL | Delta |
| SP | Southern Plains |
| MT | Mountain |
| PC | Pacific Coast |

## 4. Extensions: Nested Logit

### 4.1 Motivation

If IIA is violated (e.g., crop and pasture are closer substitutes), nested logit provides a solution by grouping similar alternatives into "nests."

### 4.2 Nesting Structure for Land Use

A natural nesting structure:
```
                    Land Use Choice
                    /              \
           Agricultural          Non-Agricultural
          /     |     \          /        \
        Crop  Pasture Range   Forest    Urban
```

### 4.3 Nested Logit Probability

The probability of choosing alternative *i* in nest *m* is:

$$P_i = P_{i|m} \cdot P_m$$

Where:
$$P_{i|m} = \frac{e^{V_i / \lambda_m}}{\sum_{j \in m} e^{V_j / \lambda_m}}$$

$$P_m = \frac{I_m^{\lambda_m}}{\sum_k I_k^{\lambda_k}}$$

And the **inclusive value** (log-sum) is:
$$I_m = \ln \sum_{j \in m} e^{V_j / \lambda_m}$$

The parameter $\lambda_m \in (0,1]$ measures correlation within the nest:
- $\lambda_m = 1$: No correlation (collapses to MNL)
- $\lambda_m < 1$: Positive correlation within nest

## 5. Interpretation of Results

### 5.1 Marginal Effects

In multinomial logit, the marginal effect of variable $x_k$ on the probability of alternative *j* is:

$$\frac{\partial P_j}{\partial x_k} = P_j \left( \beta_{jk} - \sum_{i=1}^{J} P_i \beta_{ik} \right)$$

This shows that:
- Effects depend on current probabilities (nonlinear)
- A variable can affect alternatives where it doesn't directly enter (through the denominator)

### 5.2 Elasticities

**Own-elasticity** (effect of attribute of *j* on probability of *j*):
$$\eta_{jj} = \beta_j x_j (1 - P_j)$$

**Cross-elasticity** (effect of attribute of *k* on probability of *j*):
$$\eta_{jk} = -\beta_k x_k P_k$$

### 5.3 Welfare Analysis

The expected maximum utility (log-sum) provides a welfare measure:

$$E[\max_j U_{nj}] = \ln \sum_{j=1}^{J} e^{V_{nj}} + C$$

Changes in this measure due to policy can be converted to dollar values using the coefficient on income or returns.

## 6. References

**Textbook:**
- Train, K. (2009). *Discrete Choice Methods with Simulation*, 2nd Edition. Cambridge University Press. Available at: https://eml.berkeley.edu/books/choice2.html

**Land Use Applications:**
- Lubowski, R.N., A.J. Plantinga, and R.N. Stavins (2006). "Land-use change and carbon sinks: Econometric estimation of the carbon sequestration supply function." *Journal of Environmental Economics and Management*, 51(2): 135-152.
- Lubowski, R.N., A.J. Plantinga, and R.N. Stavins (2008). "What drives land-use change in the United States? A national analysis of landowner decisions." *Land Economics*, 84(4): 529-550.
- Mihiar, C.M., and D.J. Lewis (2023). "An empirical analysis of U.S. land-use change under multiple climate change scenarios." *Journal of the Agricultural and Applied Economics Association*, 2: 597-611.

**Original Dissertation:**
- Mihiar, C. (2018). "An Econometric Analysis of the Impact of Climate Change on Forest Land Value and Broad Land-use Change." PhD Dissertation, Oregon State University.
