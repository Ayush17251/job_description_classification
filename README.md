a classifier which takes in a job description and gives the department name for it.

*   Using a neural network model
*   Making use of a pre-trained Word Embeddings (example: Word2Vec, GloVe, etc.)

### Data Structring

All the data required can be found in tha data folder in the root of the project

There are two sources for loading your training/test data

*   For *Job Description*:  
   **docs** folder contains around 1000 json files, each of which is a single job posting. You have to use the value of `description` field inside the `jd_information` field.

*   For *Job Department*:  
   **document_departments.csv** file contains the mapping of document id to department name where document id is the name of the corresponding file in docs folder.

Note: Make sure to have GoogleNews-vectors-negative300.bin in the current working directory.
