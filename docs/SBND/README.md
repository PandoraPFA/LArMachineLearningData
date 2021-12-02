# SBND Feature Tool Sets
This document records the feature sets that are used in the current (sbndcode v09_37_01) PandoraSettings XML files. There are currently 3 MVAs run in SBND: Vertexing, PFO Characterisation, and SliceID. Note, however, that there are multiple versions of each of these that are run each with different parameter sets.

# Vertexing
Two BDTs (Region then Vertex) with very similar variables. Both make comparisons between pairs of candidates and have three types of variables. Event variables describe the event as a whole and are independent of the two candidates, vertex variables describe the candidate and its surrounding hits (therefore you get 2 of each, one for each candidate) and shared variables describe links between the two candidates.
### Region
##### Event Variables
- Showeryness 
  - _the proportion of the hits in the event that are currently in "showery" clusters_
- Area 
  - _the area taken up by the event in 2D space (drift x wire)_
  - _both dimensions calculated by the central 90% of hits & averaged across all 3 views_
- Longitudinality
  - _the shape of the event, calculated as z / (x + z) where x & z are the drift direction and wire plane lengths respectively_
  - _0->1 where 0.5 is a perfectly square event_
- Number of Hits
- Number of Clusters
- Number of Vertex Candidates
##### Vertex Variables
- Beam Deweighting 
  - _a measure of how far upstream in the beam direction the candidate lies_
- Energy Kick
  - _a measure of the imbalance in transverse energy from the candidate vertex's location_
- Global Asymmetry
  - _a measure of how much of the event's energy lies infront or behind the vertex when projected onto an "event axis"_
- Local Asymmetry
  - _as global asymmetry but using only clusters in close proximity to the vertex_
- Shower Asymmetry
  - _as global asymmetry but using only clusters labelled as "showery"_
- Energy Deposition Asymmetry
  - _as global asymmetry but calculate an asymmetry in the energy deposited per unit length along the axis_
- Energy 
  - _the sum of the energies of the nearest hit to the candidate in each plane_
##### Shared Variables
- Separation
  - _the distance between the two candidates_
- Axis Hits
  - _the number of hits along the axis between the candidates, normalised by the length of the axis (separation)_

### Vertex
Same as the region BDT but with the addition of a single vertex variable...
##### Vertex Variables
- R Phi
  - _measure of the r/phi distribution of hits in the vicinity of the candidate_
  
  
# PFO Characterisation
- Length
- Straight Line Diff Mean
- Max Fit Gap length
- Sliding Linear Fit RMS
- Vertex Distance
- PCA Secondary-Primary Ratio
- PCA Tertiary-Primary Ratio
- Open Angle Difference
- Fractional Charge Spread
- Charge End Fraction
### No Charge Info
This version of the MVA is run when no collection plane information is available so the calorimetric variables (Fractional Charge Spread and Charge End Fraction) are excluded.

# SliceID
- Number of Final State PFOs
- Total Number of Hits
- Vertex Y (Vertical) Position
- Weighted Z (Beam) Direction
- Number of Space-Points in Sphere
- Eigenvalue Ratio in Sphere
- Longest Track Y (Vertical) Direction
- Longest Track Deflection
- Fraction of Hits in Longest Track
- Number of Hits in Longest Track
