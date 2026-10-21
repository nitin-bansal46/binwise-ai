# Data and rule sources

## RealWaste

- Dataset: [UCI RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste)
- DOI: `10.24432/C5SS4G`
- Size/classes: 4,752 images across nine waste categories
- Licence: CC BY 4.0
- Local handling: `scripts/download_realwaste.py`; `data/raw/` is ignored by Git

## Zurich

- Source: [City of Zurich — Correct waste disposal in Zurich](https://www.stadt-zuerich.ch/content/dam/web/de/umwelt-energie/entsorgung/dokumente/correct-waste-disposal-zurich.pdf)
- Scope: category-level preparation and disposal routes
- Verification date stored in each RuleCard: 2026-10-20

## Gurugram / India

- Local authority: [Municipal Corporation Gurugram](https://mcg.gov.in/)
- National rule source: [CPCB — Solid Waste Management Rules, 2026](https://swm.cpcb.gov.in/images/solid-waste-management-rules-2026.pdf), effective 1 April 2026
- Scope: source segregation into wet, dry, sanitary and special-care streams, with separately governed waste kept out;
  actual collection arrangements must be checked with the current ward or housing society service
- Verification date stored in each RuleCard: 2026-10-20

The corpus is a transformed summary, not a reproduction of the source documents. German and Hindi
translations are machine-assisted and should receive native-speaker review before being described
as professionally translated.

