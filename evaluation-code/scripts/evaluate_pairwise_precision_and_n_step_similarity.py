import numpy as np
import re
import argparse
import pandas as pd
import os
import pdb

'''
Inputs:
    generated_plan: list of all grounded steps in the generated plan
    ground_truth_plan: list of all gorunded steps in the gt plan

Output:
    pairwise_precision: the pairwise precision between generated and ground truth plan

Rough Operation:
    - split generated_plan into a count dictionary of all pairs of steps
    - split ground_truth_plan into a count dictionary of all pairs of steps
    - compute the precision using counts of pairs
'''
def compute_pairwise_precision(generated_plan, ground_truth_plan):

    generated_count_dictionary = {}

    for i in range(len(generated_plan)):
        for j in range(i+1, len(generated_plan)):
            
            if ';'.join([generated_plan[i], generated_plan[j]]) not in generated_count_dictionary:
                generated_count_dictionary[';'.join([generated_plan[i], generated_plan[j]])] = 0


            generated_count_dictionary[';'.join([generated_plan[i], generated_plan[j]])] += 1
    

    gt_count_dictionary = {}

    for i in range(len(ground_truth_plan)):
        for j in range(i+1, len(ground_truth_plan)):
            
            if ';'.join([ground_truth_plan[i], ground_truth_plan[j]]) not in gt_count_dictionary:
                gt_count_dictionary[';'.join([ground_truth_plan[i], ground_truth_plan[j]])] = 0


            gt_count_dictionary[';'.join([ground_truth_plan[i], ground_truth_plan[j]])] += 1
    

    precision = 0.0

    for pairing in generated_count_dictionary.keys():

        precision += min(generated_count_dictionary[pairing], gt_count_dictionary[pairing] if pairing in gt_count_dictionary else 0.0)
    

    return precision/sum(generated_count_dictionary.values()) if sum(generated_count_dictionary.values()) > 0 else 0,  precision/sum(gt_count_dictionary.values()) if sum(gt_count_dictionary.values()) > 0 else 0



'''
Inputs:
    generated_plan: list of all grounded steps in the generated plan
    ground_truth_plan: list of all gorunded steps in the gt plan

Output:
    n_step_similarity: the geometric mean of precision for 1 to n 'step window' between generated and ground truth plan

Rough Operation:
    - iterate over all step windows sizes
    - collect step window sub-plans in generated_plan and for ground_truth_plan
    - compute the precision for current step window size
    - collect all precisions and compute geometric average
'''
def compute_n_step_similarity(generated_plan, ground_truth_plan, n=3):

    generated_count_dictionary = {}
    gt_count_dictionary = {}

    n_step_precision = []
    n_step_accuracy = []

    for step in range(1,n+1):
        
        for i in range(0, len(generated_plan)-step+1):

            generated_subplan = ';'.join(generated_plan[i : i+step])
            gt_subplan = ';'.join(ground_truth_plan[i : i+step])


            if generated_subplan not in generated_count_dictionary:
                generated_count_dictionary[generated_subplan] = 0
            if gt_subplan not in gt_count_dictionary:
                gt_count_dictionary[gt_subplan] = 0
            
            generated_count_dictionary[generated_subplan] += 1
            gt_count_dictionary[gt_subplan] += 1


        precision = 0.0

        for pairing in generated_count_dictionary.keys():

            precision += min(generated_count_dictionary[pairing], gt_count_dictionary[pairing] if pairing in gt_count_dictionary else 0.0)

        precision = precision/sum(generated_count_dictionary.values()) if sum(generated_count_dictionary.values()) > 0 else 0

        accuracy = precision/sum(gt_count_dictionary.values()) if sum(gt_count_dictionary.values()) > 0 else 0


        # print('{}-step precision: {}'.format(step, precision))

        n_step_precision.append(precision)
        n_step_accuracy.append(accuracy)

    
    final_n_step_precisions = []
    #compute n-step precision for every 'k' from 1 to n
    for k in range(1, n+1):

        final_precision = np.power(np.prod(np.array(n_step_precision[:k])), 1/k)
        final_n_step_precisions.append(final_precision)
    

    final_n_step_accuracies = []
    for k in range(1, n+1):
        final_accuracy = np.power(np.prod(np.array(n_step_accuracy[:k])), 1/k)
        final_n_step_accuracies.append(final_accuracy)

    return final_n_step_precisions, final_n_step_accuracies



    
def loop_and_parse_scripts(dataframe, num =3):

   

    #replace the identity argument (xx) with empty space + split based on comma to get list of steps
    def preprocess_scripts(scripts):

        for i in range(len(scripts)):
            scripts[i] = re.sub('\((.*?)\)', '', scripts[i]) if type(scripts[i])==str else ''
        scripts = [x.split(',') for x in scripts]
        
        for i in range(len(scripts)):

            scripts[i] = [x.strip() for x in scripts[i]]

        return scripts

    # dataframe = dataframe[dataframe['scene']==2]
    print(len(dataframe))
    gt_program_scripts = dataframe['most_similar_gt_program_text'].values
    generated_program_scripts = dataframe['parsed_text'].values

    gt_program_scripts = preprocess_scripts(gt_program_scripts)
    generated_program_scripts = preprocess_scripts(generated_program_scripts)






    total_pairwise_precision = 0.0
    total_pairwise_acc = 0.0
    total_n_step_similarities = {k:0 for k in range(1, num+1)}
    total_n_step_accuracies = {k:0 for k in range(1, num+1)}

    for row in range(len(generated_program_scripts)):
        
        pairwise_precision, pairwise_acc = compute_pairwise_precision(generated_program_scripts[row], gt_program_scripts[row])
        n_step_similarities, n_step_accuracies = compute_n_step_similarity(generated_program_scripts[row], gt_program_scripts[row], n=num)

        total_pairwise_precision += pairwise_precision
        total_pairwise_acc += pairwise_acc
        
        for k in range(1,num+1):
            total_n_step_similarities[k] += n_step_similarities[k-1]
            total_n_step_accuracies[k] += n_step_accuracies[k-1]



    total_pairwise_precision /= len(generated_program_scripts)
    total_pairwise_acc /= len(generated_program_scripts)

    for key in total_n_step_similarities.keys():
        total_n_step_similarities[key] = total_n_step_similarities[key]/len(generated_program_scripts)

    for key in total_n_step_accuracies.keys():
        total_n_step_accuracies[key] = total_n_step_accuracies[key]/len(generated_program_scripts)



    print('Average Pairwise Precision: ', total_pairwise_precision) 
    print('Average Pairwise Accuracy: ', total_pairwise_acc)
    print('Average N-Step Precision: ', total_n_step_similarities)
    print('Average N-Step Accuracy: ', total_n_step_accuracies)


if __name__=='__main__':
    #load the csv file into a pandas dataframe to be processed
    parser = argparse.ArgumentParser(description='Process some integers.')
   
    parser.add_argument('--file', type=str, help='csv filepath with data to process for metrics')

    args = parser.parse_args()

    dataframe = pd.read_csv(os.path.relpath(args.file))
    pdb.set_trace()
    loop_and_parse_scripts(dataframe)