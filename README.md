data tracking link below for the Code Repository, project running log and reports

https://github.com/Stephenwjj/E3_machine-learning-in-finance


## Run the E3_project online in Colab with the belowing code

%cd /content
!rm -rf E3_machine-learning-in-finance

!git clone https://github.com/Stephenwjj/E3_machine-learning-in-finance.git
%cd E3_machine-learning-in-finance

!git config --global user.name "Stephenwjj"
!git config --global user.email "wjjdhu@126.com"

!pip install -q -r requirements.txt
!python main.py
