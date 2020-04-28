## Data Collection Pipeline:

### 1. Collecting the Data

The notebook *sample_collection.ipynb* contains all the code used to collect the samples for this experiment. 
The python file *sample_collection.py* is generated from this notebook by using Jupyter Notebook's built in export function. (this can be seen by looking at some of the artifacts in this file which look like elements of the python notebook).

The intention of including both these files is that *sample_collection.ipynb* be used by the reader to inspect and verify the sampling process, and *sample_collection.py* be used when actually collecting the data since it can be run in it's entirety with a single command. 

After running these files, sample data will be recorded as TensorBoard logs in the 

With the default parameters set, about 2GB of files will be generated. In zipped form, this can be reduced to about 200MB - furthermore transferring compressed files between computers is significantly faster than in their raw form with nearly 7,000 directories and 1,0000 files. Thus I highly recommend zipping the files before downloading them from AWS or other testng environment. 

I provide the data I used for this experiment in *data_logs.zip* to avoid making this repository unwieldly.


### 2. Exporting the Data 

The TensorBoard log format is fairly opaque in format and no decent documentation exists that I could find. I beleive that this is because the designers expect you to analyze your results using the visualizer built in to TensorBoard, which is good for studying a single network but insufficient for a data-based approach for analysis. Thankfully however, TensorBoard provides a way of exporting to "csv" (really it is a json formatted file).

I spent significant effort looking for a way to automate this process, but was unable to find any solution to this problem and was forced to export all files manually. 

This can be a very time consuming process.

I have included the exported files in (filepath)


---

After completion of these steps, data is now ready for input into the analysis pipeline.
