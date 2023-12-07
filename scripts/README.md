
## Students version generation

Students version - stripped from solutions can be generated automatically using the 
[scripts/makeStudentsVersion.py][scripts/makeStudentsVersion.py] script.
This script removes code fragments between `#BEGIN_SOLUTION` and `#END_SOLUTION` tags
and copies it to the [studentsVersions](studentsVersions) directory.
Script usage:

```Python
./scripts/makeStudentsVersion.py -i 03_Trening_modelu.ipynb
```

Space between tags will be substituted with a single line with three dots: 

``` python
def classifier (X, threshold):
    #BEGIN_SOLUTION
    return X > threshold
    #END_SOLUTION
```

The above in student's version will become:
``` python
def classifier (X, threshold):
    ...
```

**Note**:
* ```#END_SOLUTION``` tag can not be last line in a cell. If this is the case put ```pass``` statement after it:

``` python
def classifier (X, threshold):
    #BEGIN_SOLUTION
    return X > threshold
    #END_SOLUTION
pass    
```
* in the case of spelling mistake in tags the notebook will be broken - check how it looks like in GitHub