# Ambulance Relocation Optimizer — Project Idea

## The three datasets

- **Toronto Centreline (TCL)** — the road network. It's the city's flagship
  street-geometry dataset and feeds straight into cuGraph as your graph edges
  for the travel-time matrix. It shows up across the open data gallery as the
  standard base layer for routing and drive-time projects.

- **Ambulance Station Locations** — the candidate posting points, i.e. where
  idle units can wait. This is a geographic file containing the locations of
  ambulance stations within the City of Toronto, including the district
  offices. It's exactly the "where can a unit relocate to" set, and it's a
  dedicated dataset, not something you have to hand-build.

- **Paramedic Services Incident Data** — the demand layer, and it's real call
  data rather than a population guess. It covers all incidents Toronto
  Paramedic Services responds to, including incident type, priority, and number
  of units that arrived on scene.
  - **Privacy catch:** street and cross-street fields are removed; only the
    Forward Sortation Area (FSA — the first three postal-code characters) is
    given for location. So demand resolves to FSA zones, not exact points —
    which is fine, your coverage zones can just be the FSAs.
  - **Honesty flag:** one portal page showed this dataset with a "Retired"
    badge, so confirm it's still live in hour one. If it's gone, fall back to
    Neighbourhood Profiles population as the demand weight.

> The FSA detail is actually a gift — it hands you your zone definition for
> free. **Demand zones = FSAs, stations = posting candidates, centreline = the
> graph between them.**

## The split (3 people, no frontend lane, all on the hard part)

### Person 1 — Data + Graph (RAPIDS foundation)
Owns:
- Acquiring and verifying all three dataset downloads.
- cuDF cleaning of centreline, stations, and incidents.
- cuGraph building the road graph and the station↔FSA travel-time matrix, cached.
- Precomputing the coverage relation (which stations reach which FSAs within the
  response threshold).

Delivers the cached matrix + coverage relation + the zone/station/demand objects
everyone else consumes. This is the critical path and the messiest data work, so
it's a full lane on its own.

### Person 2 — Optimization brain (cuOpt + evaluation)
Owns:
- The coverage evaluator (available units + matrix → coverage map + list of gap zones).
- The Monte Carlo expected-coverage on GPU (the performance and Spark story).
- The cuOpt MILP for relocation with the move-penalty term.
- Wrapping it all as one `re-optimize(state) → moves` entry point.

This is the heaviest algorithmic lane — put your strongest technical person here,
since the rubric's depth points live here.

### Person 3 — Simulation + Agent (Nemotron + orchestration)
Owns:
- The discrete-event simulation engine (call arrivals, unit busy/free state
  machine — the thing that generates events and drives the demo).
- The local Nemotron NIM setup.
- The agent logic (plain-English command → cuOpt constraints, and solution →
  plain-English explanation).
- Integration — wiring sim → evaluator → optimizer → agent — since this lane
  sits at the center of the loop.

## Dependencies & contract

This split keeps the dependency clean: **Person 1 feeds Person 2 feeds Person 3**,
and Person 3's simulation feeds back into Person 2.

The hour-one interface contract still applies — agree the `matrix`, `unit`,
`coverage_map`, and `move` shapes so Person 2 builds on a mock matrix and
Person 3 builds on mock moves while Person 1 wrangles the real data.

## Protect the demo visual

Since you're dropping the frontend lane, you still need something on screen for
the demo, or the Value/Usability and wow-factor points suffer. Don't let it
become a fourth hidden job — use the cheapest possible path:

- **Kepler.gl** or **pydeck** can render points and a colored coverage layer
  from a dataframe with almost no code.
- Even a **folium** map that re-renders on each step gets you the
  red-gap-heals money-shot without a real frontend build.

Budget an hour for it at the end rather than pretending it's free.
