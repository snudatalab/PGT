# PGT: Accurate News Recommendation Coalescing Personal and Global Temporal Preferences
This is a code repogitory of PGT which is for a research paper of news recommender system.
This project contains a preprocess code for the dataset (Adressa, Globo), and an implementation of competitor methods.
This document includes a brief description of code, an environment setting, and an execution procedure.
Please refer the 'code description document' to find the detailed information.

## Environment Setting
This project should be executed by python 3.6.
You need to install the related python packages using the following command.
- pip install -r requirements.txt

## Project Structure
You can find the sub-directory structure of the project in this section.
The raw dataset files are not included in this repogitory because of the license issue.

- `Makefile`: All procedures of project are executed by the dependency tree in the Makefile.
- `src/`: the path of directory for the source codes.
- `data/`: the path of raw dataset files.
- `cache/`: the path of directory storing intermediate output files of the data preprocessing. the content of the cache will be generated automatically during the execution

Please download datasets to data/ folder of the project.
The content of data should be like the bellow.
#### Adressa
- Download url: http://reclab.idi.ntnu.no/dataset
- data/adressa/one_week/20170101 ... 20170107
- data/adressa/three_month/20170101 ... 20170331
#### Globo
- Download url: https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom
- data/globo/clicks: 'clicks' of globo dataset
- data/globo/articles_metadata.csv: 'articles_metadata.csv' of globo dataset

## How to execute
All codes should be executed by the Makefile
For example, you need to input the following command in the prompt to execute the pgt.
- make comp_pgt

## License
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
The authors are as follows:
- Bonhun Koo (<darkgs@snu.ac.kr>)
- Hyunsik Jeon (<jeon185@snu.ac.kr>)
- U Kang (<ukang@snu.ac.kr>) - corresponding author
