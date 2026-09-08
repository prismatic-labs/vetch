# Water sourcing worksheet

Phase 0 of provenance-gated water estimates. Candidate WUE values only;
nothing here is a registry entry until it has a retrievable citation.
Consulted 2026-08-18. All figures in liters per kWh.

## Measurement-boundary trap

The Green Grid White Paper #35 (2011) defines two metrics:

| Metric | Boundary | Formula |
|---|---|---|
| WUE (site) | On-site | annual site water usage (L) / IT energy (kWh) |
| WUE<sub>source</sub> | On-site + off-site electricity | site WUE + EWIF × PUE |

Citation: Patterson, M. et al. (2011). *Water Usage Effectiveness (WUE™): A Green Grid Data Center Sustainability Metric*, WP#35. https://www.thegreengrid.org/en/resources/library-and-tools/238-WP%2335---Water-Usage-Effectiveness-(WUE):-A-Green-Grid-Data-Center-Sustainability-Metric

A second distinction sits inside the on-site boundary and is **not** the same as site vs source:

- **Withdrawal** — volume drawn from municipal/surface/ground sources.
- **Consumption** — withdrawal minus water returned; typically evaporation plus blowdown.

AWS publishes withdrawal. Google publishes ISO WUE Category 2 (consumptive: input minus returned). Microsoft publishes “liters used for humidification and cooling” (Green Grid site WUE). Mixing these in a comparison is wrong in a way no uncertainty band communicates. Registry `boundary` is `onsite` for every surviving entry; withdrawal vs consumption is recorded in `basis` and must not be collapsed.

## Fleet-average vs per-region trap

Google publishes campus-level **water volumes** and campus-level **PUE**, but not campus-level energy (kWh) in a form that lines up with those water volumes, and not campus-level WUE. Deriving per-region WUE would require inventing an energy denominator. That is not done here.

Microsoft publishes continental WUE (Americas / Asia Pacific / EMEA) plus a global fleet figure. No per-Azure-region WUE.

AWS publishes global, geography, **and** per-AWS-region WUE for regions with ≥12 months of data.

## Jegham et al. 2025 (arXiv:2505.09598v6)

Jegham’s water figures are **not independent**. Equation (4) is:

`Water (L) = (E_query / PUE) · WUE_site + E_query · WUE_source`

They assign Azure-reported or AWS-reported site WUE to hosted models and add a source-level (grid) term. Using Jegham water as a registry value would double-count publisher WUE and mix the source boundary we are not using. Not entered.

## Unsourceable provider keys (confirmed)

| key | Finding |
|---|---|
| `openai` | No facility WUE. Stanford FMTI 2025 scores water usage for the final training run at 0. Sam Altman’s 2025 per-query water claim (0.000085 gal) is a product-level anecdote, not a facility WUE, and cannot be converted without inventing energy and a facility. Drop. |
| `anthropic` | No sustainability report and no water disclosure as of 2026. Does not operate disclosed-WUE facilities of its own; serving is split across AWS, Google Cloud, and other partners, so even a host-provider WUE cannot be applied without knowing which host served the call. Drop. |
| `vertexai` | Vertex AI does not publish a separate WUE. It runs on Google Cloud data centers, which **do** publish WUE (see `google` / `vertexai` rows below). The key is kept only as an alias of Google’s published fleet figure, with that attribution in `basis`. |
| `deepseek` | No publisher WUE located. Not entered. |

## Candidate values

| key | value_l_per_kwh | boundary | published_or_derived | derivation | publisher | document | year | url | section | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| google | 1.15 | onsite | published | — | Google | Measuring the environmental impact of delivering AI at Google Scale | 2025 | https://arxiv.org/abs/2508.15734 | Water consumption methodology; ISO WUE Category 2 | Consumptive (input − returned). Calendar-year average of data centers supporting LLM models; 1.15 L/kWh in both 2023 and 2024. Not a per-region figure. Google’s Environmental Report publishes PUE and total water, not fleet WUE; the AI paper is the WUE primary. |
| vertexai | 1.15 | onsite | published | same figure as `google` | Google | Measuring the environmental impact of delivering AI at Google Scale | 2025 | https://arxiv.org/abs/2508.15734 | same | Vertex AI is a Google Cloud product. Not independently disclosed. |
| aws | 0.12 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page (PUE and WUE tables) | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Water usage effectiveness; Global row | Withdrawal WUE: liters withdrawn per kWh of IT load. Calendar year 2025. Also stated in 2025 AWS Summary PDF. |
| aws-us-east-1 | 0.06 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | AWS Region PUE and WUE; U.S. East (Northern Virginia) | Withdrawal. |
| aws-us-east-2 | 0.06 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | U.S. East (Ohio) | Withdrawal. |
| aws-us-west-1 | 0.39 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | U.S. West (Northern California) | Withdrawal. |
| aws-us-west-2 | 0.12 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | U.S. West (Oregon) | Withdrawal. |
| aws-ca-central-1 | 0.06 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Canada (Central) | Withdrawal. |
| aws-ca-west-1 | 0.04 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Canada (West) | Withdrawal. |
| aws-sa-east-1 | 0.09 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | South America (Sao Paulo) | Withdrawal. |
| aws-eu-central-1 | 0.17 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Europe (Frankfurt) | Withdrawal. |
| aws-eu-west-1 | 0.02 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Europe (Ireland) | Withdrawal. |
| aws-eu-west-2 | 0.14 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Europe (London) | Withdrawal. |
| aws-eu-south-2 | 0.12 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Europe (Spain) | Withdrawal. |
| aws-eu-north-1 | 0.02 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Europe (Stockholm) | Withdrawal. |
| aws-ap-southeast-1 | 1.57 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Asia-Pacific (Singapore) | Withdrawal. |
| aws-ap-southeast-2 | 0.1 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Asia-Pacific (Sydney) | Withdrawal. |
| aws-ap-southeast-3 | 2.85 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Asia-Pacific (Jakarta) | Withdrawal. |
| aws-ap-southeast-4 | 0.06 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Asia-Pacific (Melbourne) | Withdrawal. |
| aws-ap-northeast-1 | 1.22 | onsite | published | — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Asia-Pacific (Tokyo) | Withdrawal. |
| aws-eu-west-3 | 0.05 | onsite | derived | geography Europe WUE 0.05 applied to Paris; AWS listed the region with PUE but WUE as — | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Geography row Europe; Europe (Paris) has no per-region WUE | Withdrawal. Arithmetic: value = published Europe geography WUE. |
| aws-ap-south-1 | 1.1 | onsite | derived | geography Asia Pacific (excl. China) WUE 1.1 applied to Mumbai | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Geography row Asia Pacific (excl. China); Asia-Pacific (Mumbai) WUE — | Withdrawal. Arithmetic: value = published APAC geography WUE. |
| aws-ap-south-2 | 1.1 | onsite | derived | geography Asia Pacific (excl. China) WUE 1.1 applied to Hyderabad | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Geography row Asia Pacific (excl. China); Asia-Pacific (Hyderabad) WUE — | Withdrawal. |
| aws-ap-northeast-3 | 1.1 | onsite | derived | geography Asia Pacific (excl. China) WUE 1.1 applied to Osaka | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Geography row Asia Pacific (excl. China); Asia-Pacific (Osaka) WUE — | Withdrawal. |
| bedrock | 0.12 | onsite | published | same figure as `aws` | Amazon Web Services | AWS Cloud sustainability page | 2025 | https://sustainability.aboutamazon.com/products-services/aws-cloud | Global row | Bedrock is an AWS service. Not independently disclosed. |
| azure | 0.27 | onsite | published | — | Microsoft | Measuring energy and water efficiency for Microsoft datacenters | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | WUE table, Global row | Site WUE: annual liters used for humidification and cooling / IT kWh. FY25 = 1 Jul 2024–30 Jun 2025. Owned-and-operated sites operational 12 months; leased excluded. |
| azure-eastus | 0.34 | onsite | published | continental Americas WUE applied to eastus | Microsoft | Measuring energy and water efficiency for Microsoft datacenters | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | WUE table, Americas row | No per-region WUE published. Arithmetic: value = published Americas WUE. |
| azure-eastus2 | 0.34 | onsite | published | continental Americas WUE applied to eastus2 | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Americas row | same |
| azure-westus | 0.34 | onsite | published | continental Americas WUE applied to westus | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Americas row | same |
| azure-westus2 | 0.34 | onsite | published | continental Americas WUE applied to westus2 | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Americas row | same |
| azure-centralus | 0.34 | onsite | published | continental Americas WUE applied to centralus | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Americas row | same |
| azure-northeurope | 0.03 | onsite | published | continental EMEA WUE applied to northeurope | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Europe, Middle East & Africa row | same |
| azure-westeurope | 0.03 | onsite | published | continental EMEA WUE applied to westeurope | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Europe, Middle East & Africa row | same |
| azure-japaneast | 0.25 | onsite | published | continental Asia Pacific WUE applied to japaneast | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Asia Pacific row | same |
| azure-southeastasia | 0.25 | onsite | published | continental Asia Pacific WUE applied to southeastasia | Microsoft | same | FY25 | https://datacenters.microsoft.com/sustainability/efficiency/ | Asia Pacific row | same |

## Explicitly not entered

| Candidate | Why not |
|---|---|
| `DEFAULT_WUE` 1.8 | Industry-average guess. No publisher, no document for “the” 1.8 L/kWh that Vetch was using. Green Grid WP#35 cites 1.8 L/kWh as a **US national average EWIF** (source-boundary electricity water), not on-site datacenter WUE. Using it as a site default is a category error. |
| Previous `wue.json` per-GCP-region numbers (0.8–2.8) | No primary document found that publishes those values. Google does not publish per-facility WUE. Campus water volumes cannot be converted without campus IT energy, which Google does not publish in an aligned table. |
| Previous `wue.json` per-AWS-region numbers that disagree with the 2025 table (e.g. `aws-us-east-1` was 0.8; AWS publishes 0.06) | Replaced by the 2025 published figures. The old numbers were unsourced. |
| Previous `wue.json` per-Azure-region numbers (0.85–3.2) | Microsoft does not publish per-region WUE. Replaced by continental averages where a continent is published. |
| AWS Middle East, Africa, China region keys | Geography WUE listed as N/A. A missing entry produces `None`. |
| Google per-campus WUE derived from water gallons ÷ assumed energy | Energy denominator is not published per campus. Derivation would invent an input. |
| OpenAI / Anthropic | Confirmed unsourceable. See table above. |
| Source-boundary (WUE<sub>source</sub>) values | Publishers above report on-site metrics. Mixing source into the same registry would make cross-provider comparisons invalid. |

## Registry mapping

- **Tier 1** — AWS per-region rows whose 2025 WUE cell is a number.
- **Tier 2** — unused. No water-volume ÷ energy derivation was possible with both inputs published.
- **Tier 3** — fleet or geography/continental averages, including those averages applied to a named region (`azure-eastus`, `aws-ap-south-1`, provider keys `google` / `aws` / `azure`).
