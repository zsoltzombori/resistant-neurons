# Here are the results.

## Baseline if I don't hammer the neurons: 0.874
```
  100:  dev acc 0.850
  120:    train acc 0.868    dev acc 0.845
  140:    train acc 0.838    dev acc 0.855
  160:    train acc 0.854    dev acc 0.848
  180:    train acc 0.870    dev acc 0.874
  200:    train acc 0.862    dev acc 0.862
  220:    train acc 0.870    dev acc 0.860
  240:    train acc 0.856    dev acc 0.857
  260:    train acc 0.868    dev acc 0.869
  280:    train acc 0.866    dev acc 0.873
  300:    train acc 0.892    dev acc 0.866
```
## Each neuron randomized: 0.851
```
  100:  dev acc 0.077
  120:    train acc 0.640    dev acc 0.665
  140:    train acc 0.724    dev acc 0.732
  160:    train acc 0.762    dev acc 0.780
  180:    train acc 0.828    dev acc 0.803
  200:    train acc 0.806    dev acc 0.796
  220:    train acc 0.832    dev acc 0.831
  240:    train acc 0.810    dev acc 0.836
  260:    train acc 0.866    dev acc 0.844
  280:    train acc 0.834    dev acc 0.851
  300:    train acc 0.844    dev acc 0.839
```

## Only bad neurons are randomized, 50 percentile: 0.862
```
  100:  dev acc 0.344
  120:    train acc 0.786    dev acc 0.778
  140:    train acc 0.820    dev acc 0.809
  160:    train acc 0.822    dev acc 0.821
  180:    train acc 0.834    dev acc 0.839
  200:    train acc 0.838    dev acc 0.840
  220:    train acc 0.876    dev acc 0.829
  240:    train acc 0.854    dev acc 0.855
  260:    train acc 0.858    dev acc 0.858
  280:    train acc 0.824    dev acc 0.862
  300:    train acc 0.850    dev acc 0.848
```
## Only bad neurons are randomized, 20 percentile: 0.881
```
  100:  dev acc 0.707
  120:    train acc 0.856    dev acc 0.837
  140:    train acc 0.848    dev acc 0.848
  160:    train acc 0.830    dev acc 0.852
  180:    train acc 0.876    dev acc 0.866
  200:    train acc 0.826    dev acc 0.846
  220:    train acc 0.848    dev acc 0.859
  240:    train acc 0.896    dev acc 0.866
  260:    train acc 0.848    dev acc 0.855
  280:    train acc 0.890    dev acc 0.881
  300:    train acc 0.854    dev acc 0.862
```

## Only bad neurons are randomized, 10 percentile: 0.875

```
  100:  dev acc 0.798
  120:    train acc 0.816    dev acc 0.845
  140:    train acc 0.864    dev acc 0.848
  160:    train acc 0.872    dev acc 0.856
  180:    train acc 0.826    dev acc 0.850
  200:    train acc 0.832    dev acc 0.841
  220:    train acc 0.880    dev acc 0.872
  240:    train acc 0.900    dev acc 0.868
  260:    train acc 0.882    dev acc 0.875
  280:    train acc 0.870    dev acc 0.869
  300:    train acc 0.886    dev acc 0.862
```

## Resetting bottom 50 after every 50th iteration:
```
  101:  dev acc 0.187
  110:    train acc 0.634    dev acc 0.677
  120:    train acc 0.768    dev acc 0.770
  130:    train acc 0.802    dev acc 0.794
  140:    train acc 0.814    dev acc 0.818
  150:    train acc 0.860    dev acc 0.818
  151:  dev acc 0.356
  160:    train acc 0.702    dev acc 0.711
  170:    train acc 0.774    dev acc 0.785
  180:    train acc 0.786    dev acc 0.800
  190:    train acc 0.832    dev acc 0.821
  200:    train acc 0.852    dev acc 0.820
  201:  dev acc 0.204
  210:    train acc 0.564    dev acc 0.604
  220:    train acc 0.744    dev acc 0.781
  230:    train acc 0.818    dev acc 0.798
  240:    train acc 0.824    dev acc 0.816
  250:    train acc 0.836    dev acc 0.826
  251:  dev acc 0.392
  260:    train acc 0.734    dev acc 0.732
  270:    train acc 0.802    dev acc 0.778
  280:    train acc 0.820    dev acc 0.803
  290:    train acc 0.844    dev acc 0.821
  300:    train acc 0.840    dev acc 0.838
```
## Resetting bottom 25 after every 50 th:
```
  101:  dev acc 0.729
  110:    train acc 0.764    dev acc 0.796
  120:    train acc 0.846    dev acc 0.821
  130:    train acc 0.876    dev acc 0.835
  140:    train acc 0.814    dev acc 0.848
  150:    train acc 0.828    dev acc 0.851
  151:  dev acc 0.780
  160:    train acc 0.846    dev acc 0.830
  170:    train acc 0.866    dev acc 0.819
  180:    train acc 0.828    dev acc 0.843
  190:    train acc 0.854    dev acc 0.841
  200:    train acc 0.854    dev acc 0.848
  201:  dev acc 0.791
  210:    train acc 0.830    dev acc 0.833
  220:    train acc 0.830    dev acc 0.842
  230:    train acc 0.864    dev acc 0.855
  240:    train acc 0.860    dev acc 0.858
  250:    train acc 0.856    dev acc 0.855
  251:  dev acc 0.734
  260:    train acc 0.848    dev acc 0.819
  270:    train acc 0.836    dev acc 0.830
  280:    train acc 0.848    dev acc 0.839
  290:    train acc 0.842    dev acc 0.852
  300:    train acc 0.854    dev acc 0.862
```