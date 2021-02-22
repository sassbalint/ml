## understanding numpy axes

I managed to understand how numpy axes work from this wording:

_axis 0 is the outermost axis indexing the largest subarrays
 while axis n-1 is the innermost axis indexing individual elements._

Or put it another way:

```python
x.sum(axis=0) is x[sum][*][*] ~ just do,
x.sum(axis=1) is x[*][sum][*] ~ take all and do,
x.sum(axis=2) is x[*][*][sum] ~ take all and take all and do.
```

cf. https://stackoverflow.com/questions/24281263

