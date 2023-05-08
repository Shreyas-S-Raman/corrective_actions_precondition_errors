from prompt_generator import PromptContext
import numpy as np
import pdb

class IncontextReprompter(PromptContext):

    def __init__(self, prompt_args, generation_params, percent_terminate, engine, max_iters, api_retry_if_failed):

        self.num_examples = prompt_args['num_examples']
        generation_params['stop']='\n'
        self.generation_params = generation_params
        self.percent_terminate = percent_terminate
        self.engine = engine
        self.max_iters = max_iters
        self.api_retry_if_failed = api_retry_if_failed

        self.default_error_message = prompt_args['default_error']

        super(IncontextReprompter, self).__init__(prompt_args['chosen_context'])

    def add_incontext_examples(self, plan_text, sentence_model, corrections_example_embedding, corrections_example_paths, device, top_k_similar, curr_step):

        contextualized_text = self.context_transformation(plan_text)

        try: 
            target_error_step = contextualized_text.split('\n')[-2].split(':')[1].strip().lower()
        except:
            pdb.set_trace()
        target_task = contextualized_text.split('\n')[0].split(':')[1].strip().lower()
        target_plan = contextualized_text.strip()         

        most_similar_example_idxs, ____ = top_k_similar(sentence_model, target_plan, corrections_example_embedding, device, top_k=self.num_examples)
        
        # np.random.shuffle(most_similar_example_idxs)

        for id in reversed(most_similar_example_idxs):

            example_correction = self.load_txt(corrections_example_paths[id]).split('\n\n')[1]

            #contextualized_text =  example_correction + '\n'+'-'*20+'\n' + contextualized_text
            contextualized_text = example_correction + '\n\n' + contextualized_text

        return contextualized_text
    

    def _generate_text(self, text, type):
        '''tracks all options for generated text + translated actions for the current step'''
        
        curr_generated = []
        curr_logprobs = []

        # query api ===================================
        self.generation_params['prompt'] = text
        if isinstance(self.engine, str):
            response = self.api_retry_if_failed(self.generation_params, max_iters=self.max_iters, engine=self.engine)
        else:
            response = self.engine(self.generation_params)

        for i in range(self.generation_params['n']):
            generated_text = response['choices'][i]['text']
            logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])


            if type=='error_object':
                generated_text = generated_text.strip().lower()
            else:
                generated_text = generated_text.strip()

            curr_generated.append(generated_text)
            curr_logprobs.append(logprob)

        
        num_to_look_at = int(self.percent_terminate * self.generation_params['n'])
        highest_ids = np.argsort(curr_logprobs)[-num_to_look_at:]

        nogen_terminate = True
        for idx in highest_ids:
            if len(curr_generated[idx]) > 0:
                nogen_terminate = False
                break
        
        best_idx = np.argsort(curr_logprobs)[-1]


        return curr_generated[best_idx], nogen_terminate or len(curr_generated[best_idx]) == 0


    

    def generate_error(self, ongoing_text, sentence_model, corrections_example_embedding, corrections_example_paths, device, top_k_similar, curr_step):

        ongoing_text_with_examples = self.add_incontext_examples(ongoing_text, sentence_model, corrections_example_embedding, corrections_example_paths, device, top_k_similar, curr_step)

        #step 1: generate error object
        ongoing_text_with_examples = ongoing_text_with_examples + 'Error Object:'
        error_obj, no_gen = self._generate_text(ongoing_text_with_examples, 'error_object')

        ongoing_text_with_examples  = ongoing_text_with_examples + ' ' + error_obj if not no_gen else ongoing_text + ' N/A'

        #step 2: generate error type
        ongoing_text_with_examples += '\nError Type:'
        error_type, no_gen = self._generate_text(ongoing_text_with_examples, 'error_type')

        ongoing_text_with_examples  = ongoing_text_with_examples + ' ' + error_type if not no_gen else ongoing_text_with_examples + ' N/A'

        #step 3: check if skip or not
        ongoing_text_with_examples += '\nSkip:'
        error_skip, no_gen = self._generate_text(ongoing_text_with_examples, 'skip')

        ongoing_text_with_examples  = ongoing_text_with_examples + ' ' + error_skip if not no_gen else ongoing_text_with_examples + ' False'

        if error_skip == 'True':
            ongoing_text += 'Step {}:'.format(curr_step+2)
            return None, ongoing_text, True


        #step 4: generate error information
        ongoing_text_with_examples += '\nError:'
        
        error_prompt, no_gen = self._generate_text(ongoing_text_with_examples, 'error_prompt')
        error_prompt = error_prompt.split('.')[0] + '. A correct step would be to'
        ongoing_text_with_examples = ongoing_text_with_examples + ' ' + error_prompt if not no_gen else ongoing_text_with_examples + ' ' + self.default_error_message

        ongoing_text += 'Error: ' + error_prompt if not no_gen else 'Error: ' + self.default_error_message
        ongoing_text += '\nStep {}:'.format(curr_step+1)
        
        ongoing_text_with_examples += '\nStep {}:'.format(curr_step+1)

        return 'Error: ' + error_prompt, ongoing_text_with_examples, False
        
        
        
