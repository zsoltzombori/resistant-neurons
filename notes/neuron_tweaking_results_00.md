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

## Cloning and noising with 50th percentile: 
```
  100:  dev acc 0.120
  120:    train acc 0.616    dev acc 0.644
  140:    train acc 0.752    dev acc 0.724
  160:    train acc 0.746    dev acc 0.749
  180:    train acc 0.800    dev acc 0.779
  200:    train acc 0.790    dev acc 0.781
  220:    train acc 0.782    dev acc 0.794
  240:    train acc 0.818    dev acc 0.808
  260:    train acc 0.806    dev acc 0.815
  280:    train acc 0.812    dev acc 0.818
  300:    train acc 0.826    dev acc 0.815
```

## Cloning and noising with 20th percentile: 0.871
```
  100:  dev acc 0.360
  120:    train acc 0.800    dev acc 0.809
  140:    train acc 0.858    dev acc 0.812
  160:    train acc 0.826    dev acc 0.837
  180:    train acc 0.868    dev acc 0.837
  200:    train acc 0.824    dev acc 0.845
  220:    train acc 0.896    dev acc 0.847
  240:    train acc 0.868    dev acc 0.858
  260:    train acc 0.884    dev acc 0.871
  280:    train acc 0.842    dev acc 0.852
  300:    train acc 0.864    dev acc 0.856
```

## Crossing good and bad, 50th percentile: 0.795

```
  100:  dev acc 0.101
  120:    train acc 0.666    dev acc 0.616
  140:    train acc 0.724    dev acc 0.749
  160:    train acc 0.764    dev acc 0.772
  180:    train acc 0.804    dev acc 0.774
  200:    train acc 0.804    dev acc 0.769
  220:    train acc 0.812    dev acc 0.785
  240:    train acc 0.808    dev acc 0.790
  260:    train acc 0.836    dev acc 0.780
  280:    train acc 0.784    dev acc 0.795
  300:    train acc 0.810    dev acc 0.795
```

## Crossing good and bad, 20th percentile: 0.874
```
  100:  dev acc 0.276
  120:    train acc 0.806    dev acc 0.790
  140:    train acc 0.804    dev acc 0.830
  160:    train acc 0.830    dev acc 0.845
  180:    train acc 0.860    dev acc 0.852
  200:    train acc 0.878    dev acc 0.865
  220:    train acc 0.874    dev acc 0.869
  240:    train acc 0.852    dev acc 0.864
  260:    train acc 0.870    dev acc 0.874
  280:    train acc 0.888    dev acc 0.862
  300:    train acc 0.862    dev acc 0.865
```

## Replacing bad with good+good 20th perc: 0.868
```
  100:  dev acc 0.303
  120:    train acc 0.770    dev acc 0.782
  140:    train acc 0.812    dev acc 0.821
  160:    train acc 0.848    dev acc 0.820
  180:    train acc 0.832    dev acc 0.839
  200:    train acc 0.842    dev acc 0.849
  220:    train acc 0.844    dev acc 0.844
  240:    train acc 0.898    dev acc 0.849
  260:    train acc 0.864    dev acc 0.868
  280:    train acc 0.854    dev acc 0.868
  300:    train acc 0.874    dev acc 0.862
```

## Replacing bad with good+good 50th perc: 0.806

```  100:  dev acc 0.085
  120:    train acc 0.464    dev acc 0.432
  140:    train acc 0.632    dev acc 0.635
  160:    train acc 0.660    dev acc 0.706
  180:    train acc 0.744    dev acc 0.735
  200:    train acc 0.756    dev acc 0.756
  220:    train acc 0.778    dev acc 0.777
  240:    train acc 0.798    dev acc 0.781
  260:    train acc 0.794    dev acc 0.790
  280:    train acc 0.808    dev acc 0.795
  300:    train acc 0.774    dev acc 0.806```