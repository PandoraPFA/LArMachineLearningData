import math

DETECTORS = ("duneFDHD1x2x6", "duneFDHD10kt", "duneFDVD1x8x6", "pduneHD")

# ATTN I have been using the scalefactor for the 3D polar_r so it will be a bit too low...
# This doesnt really matter though, there is nothing special about being bounded [0, 1] exactly, [0, a bit less than 1] is fine.
# NOTE The Cartesian x coord is always the drift direction thanks to larsoft
SCALING_FACTORS = { 
    # From *_event_data TTree of the training data ROOT file
    "duneFDHD1x2x6:polar_r"     : 0.0005044863,
    "duneFDHD1x2x6:cartesian_x" : 0.001378694,
    "duneFDHD1x2x6:cartesian_z" : 0.0007171852,
    "pduneHD:polar_r"     : 0.0009475201,
    "pduneHD:cartesian_x" : 0.001362676,
    "pduneHD:cartesian_z" : 0.002156474,
    "duneFDHD10kt:polar_r"     : 0.0001634431,
    "duneFDHD10kt:cartesian_x" : 0.0006708581,
    "duneFDHD10kt:cartesian_z" : 0.0001721244,
    "duneFDVD1x8x6:polar_r"     : 0.0005742559,
    "duneFDVD1x8x6:cartesian_x" : 0.001538462,
    "duneFDVD1x8x6:cartesian_z" : 0.00111667,
} 
assert { k.split(":")[0] for k in SCALING_FACTORS.keys() } == set(DETECTORS)
# For backcompat
SCALING_FACTORS["polar_r"] = (
    1 / math.sqrt((362.622 - -362.622)**2 + (1393.46 - -0.876221)**2 + (603.924 - -603.924)**2)
)
SCALING_FACTORS["cartesian_x"] = 1 / (362.622 - -362.622)
SCALING_FACTORS["cartesian_z"] = 1 / (1393.46 - -0.876221)

PITCHES = { 
    # From *_view_data TTree of the training data ROOT file
    "duneFDHD1x2x6" : { 4 : 0.4667, 5 : 0.4667, 6 : 0.479  },
    "pduneHD"       : { 4 : 0.4669, 5 : 0.4669, 6 : 0.4792 },
    "duneFDVD1x8x6" : { 4 : 0.765,  5 : 0.765,  6 : 0.51   } 
}
PITCHES["duneFDHD10kt"]  = { view : pitch for view, pitch in PITCHES["duneFDHD1x2x6"].items() }
assert set(PITCHES.keys()) == set(DETECTORS)
