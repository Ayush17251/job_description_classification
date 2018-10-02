## Spotmentor Assignment

### Your Task

Make a classifier which takes in a job description and gives the department name for it.

*   Use a neural network model
*   Make use of a pre-trained Word Embeddings (example: Word2Vec, GloVe, etc.)
*   Calculate the accuracy on a test set (data not used to train the model)

### Data Structring

All the data required can be found in tha data folder in the root of the project

There are two sources for loading your training/test data

*   For *Job Description*:  
   **docs** folder contains around 1000 json files, each of which is a single job posting. You have to use the value of `description` field inside the `jd_information` field.

*   For *Job Department*:  
   **document_departments.csv** file contains the mapping of document id to department name where document id is the name of the corresponding file in docs folder.

### Coding Guidelines

**Mandatory**

*   Write clean code with precise comments wherever necessary
*   Entrypoint to the code base should be run.py as already present (We should only have to execute `python run.py` to test your code)
*   Update the requirements.txt with all the packages used in your code base

**Preferable**

*   Adhere to PEP8 formatting
*   Use python 3
