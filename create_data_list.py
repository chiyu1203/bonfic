import os
import argparse
parser=argparse.ArgumentParser(description='trim video into short piece')
parser.add_argument('-i','--input_path',help='path to the video as a string: eg. C:/Users/../, apostrophe not shown',type=str,default='Z:/DATA/experiment_trackball_Optomotor')
parser.add_argument('-n','--exp_name',help='the stimulus name of the experiment: eg. xxx, apostrophe not shown',type=str,default='gratings')
parser.add_argument('-d','--exp_date',help='the date of the experiment',type=list,default=['220712','220714'])
args= parser.parse_args()
def create_data_list(input_path, exp_name, exp_date):
    data_list=[]
    #input_path = 'Z:/DATA/experiment_trackball_Optomotor'
    dataset=os.listdir(input_path)
    #exp_date=['220712','220714']
    #exp_name='gratings'
    for locust in dataset:
        locust_directory=input_path + '/' + locust
        file_list = os.listdir(locust_directory)
        for thisDate in file_list:
            if thisDate in exp_date:
                experiment_directory=locust_directory + '/' + thisDate
                experiment_list=os.listdir(experiment_directory)
                for thisExperiment in experiment_list:
                    if thisExperiment in exp_name:
                        data_list.append(experiment_directory + '/' + exp_name)
    return data_list

if __name__ == '__main__':
    create_data_list(args.input_path,args.exp_name,args.exp_date)