# Label schema and scripts

The labels are stored in straightforward json files.
Each file contains a dictionary with based on image names, e.g.,
```
['./bluefox_2016-09-30-15-19-35_bag/1475274353.721849.png',
 './bluefox_2016-11-01-10-20-23_bag/1478020942.550910.png',
 './bluefox_2016-10-10-18-41-33_bag/1476150739.595130.png']
```

For each image, there is a brief comment on possible flaws and a list of annotated vehicles.
For each vehicle, there is an axis-aligned bounding box (AABB), rear box, and side polygon.
Vehicles may not have a visible side or rear. In that case, it is marked with None.

Sample (incomplete) annotation:
```
{'flaws': ['no-issues'],
 'vehicles': [{'AABB': {'x1': 1227.86,
                        'x2': 1245.7099999999998,
                        'y1': 1043.64,
                        'y2': 1058.25},
               'rear': {'x1': 1227.86,
                        'x2': 1245.7099999999998,
                        'y1': 1043.64,
                        'y2': 1058.25},
               'side': None},
              {'AABB': {'x1': 1520.23,
                        'x2': 1572.98,
                        'y1': 1035.8,
                        'y2': 1073.78},
               'rear': {'x1': 1534.59,
                        'x2': 1572.98,
                        'y1': 1035.8,
                        'y2': 1073.78},
               'side': {'p0': {'x': 1534.5899658203125, 'y': 1035.800048828125},
                        'p1': {'x': 1534.5899658203125, 'y': 1073.780029296875},
                        'p2': {'x': 1520.22998046875, 'y': 1035.800048828125},
                        'p3': {'x': 1520.22998046875, 'y': 1071.300048828125}}},
              {'AABB': {'x1': 0.0, 'x2': 374.04, 'y1': 1058.25, 'y2': 1668.13},
               'rear': None,
               'side': {'p0': {'x': 0.0, 'y': 1058.25},
                        'p1': {'x': 0.0, 'y': 1668.1300048828125},
                        'p2': {'x': 374.0400085449219, 'y': 1059.3299560546875},
                        'p3': {'x': 374.0400085449219, 'y': 1551.8299560546875}}}]}
```
