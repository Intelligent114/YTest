import json
import logging
import math
import numpy as np
import os.path
import random
import re
from multiprocessing import Pool, Manager
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
from autogen import ConversableAgent, GroupChat, GroupChatManager, config_list_from_json
from datasets import load_dataset
from sklearn.metrics import log_loss
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(message)s')

get_json = lambda x: re.match('(.|\n)*(\{(?:[^{}]|\{.*?\})*\})(.|\n)*', x)
strip_json = lambda x:   json.loads(x.strip('```json\n').strip('```'))

country_to_nation = {
    "Albania":  "Albanian",
    "Andorra":  "Andorran",
    "Angola":  "Angolan",
    "Angola (Non-national sample)":  "Angolan",
    "Argentina":  "Argentinian",
    "Armenia":  "Armenian",
    "Australia":  "Australian",
    "Austria":  "Austrian",
    "Azerbaijan":  "Azerbaijani",
    "Bangladesh":  "Bangladeshi",
    "Bangladesh (Non-national sample)":  "Bangladeshi",
    "Belarus":  "Belarusian",
    "Belgium":  "Belgian",
    "Bolivia":  "Bolivian",
    "Bolivia (Non-national sample)":  "Bolivian",
    "Bosnia Herzegovina":  "Bosnian Herzegovinian",
    "Brazil":  "Brazilian",
    "Brazil (Non-national sample)":  "Brazilian",
    "Britain":  "British",
    "Bulgaria":  "Bulgarian",
    "Burkina Faso":  "Burkinabe",
    "Canada":  "Canadian",
    "Chile":  "Chilean",
    "China":  "Chinese",
    "China (Non-national sample)":  "Chinese",
    "Colombia":  "Colombian",
    "Colombia (Non-national sample)":  "Colombian",
    "Croatia":  "Croatian",
    "Cyprus":  "Cypriot",
    "Czech Rep.":  "Czech",
    "Czechia":  "Czech",
    "Denmark":  "Danish",
    "Ecuador":  "Ecuadorean",
    "Egypt":  "Egyptian",
    "Egypt (Non-national sample)":  "Egyptian",
    "El Salvador":  "Salvadoran",
    "Estonia":  "Estonian",
    "Ethiopia":  "Ethiopian",
    "Ethiopia (Non-national sample)":  "Ethiopian",
    "Finland":  "Finnish",
    "France":  "French",
    "Georgia":  "Georgian",
    "Germany":  "German",
    "Ghana":  "Ghanaifan",
    "Great Britain":  "British",
    "Greece":  "Greek",
    "Guatemala":  "Guatemalan",
    "Guatemala (Non-national sample)":  "Guatemalan",
    "Honduras":  "Honduran",
    "Honduras (Non-national sample)":  "Honduran",
    "Hong Kong SAR":  "Hong Konger",
    "Hungary":  "Hungarian",
    "Iceland":  "Icelander",
    "India (Current national sample)":  "Indian",
    "India (Non-national sample)":  "Indian",
    "India (Old national sample)":  "Indian",
    "Indonesia":  "Indonesian",
    "Indonesia (Non-national sample)":  "Indonesian",
    "Iran":  "Iranian",
    "Iraq":  "Iraqi",
    "Israel":  "Israeli",
    "Italy":  "Italian",
    "Ivory Coast":  "Ivorian",
    "Ivory Coast (Non-national sample)":  "Ivorian",
    "Japan":  "Japanese",
    "Jordan":  "Jordanian",
    "Jordan (Non-national sample)":  "Jordanian",
    "Kazakhstan":  "Kazakhstani",
    "Kenya":  "Kenyan",
    "Kuwait":  "Kuwaiti",
    "Kyrgyzstan":  "Kyrgyzstani",
    "Latvia":  "Latvian",
    "Lebanon":  "Lebanese",
    "Libya":  "Libyan",
    "Lithuania":  "Lithuanian",
    "Macau SAR":  "Macanese",
    "Malaysia":  "Malaysian",
    "Maldives":  "Maldivian",
    "Mali":  "Malian",
    "Mali (Non-national sample)":  "Malian",
    "Mexico":  "Mexican",
    "Mongolia":  "Mongolian",
    "Montenegro":  "Montenegrin",
    "Morocco":  "Moroccan",
    "Morocco (Non-national sample)":  "Moroccan",
    "Myanmar":  "Myanmar",
    "Netherlands":  "Dutch",
    "New Zealand":  "New Zealander",
    "Nicaragua":  "Nicaraguan",
    "Nigeria":  "Nigerian",
    "Nigeria (Non-national sample)":  "Nigerian",
    "North Macedonia":  "Macedonian",
    "Northern Ireland":  "Northern Irish",
    "Norway":  "Norwegian",
    "Pakistan":  "Pakistani",
    "Pakistan (Non-national sample)":  "Pakistani",
    "Palest. ter.":  "Palestinian",
    "Peru":  "Peruvian",
    "Philippines":  "Filipino",
    "Philippines (Non-national sample)":  "Filipino",
    "Poland":  "Polish",
    "Poland (Non-national sample)":  "Polish",
    "Portugal":  "Portuguese",
    "Puerto Rico":  "Puerto Rican",
    "Romania":  "Romanian",
    "Russia":  "Russian",
    "Russia (Non-national sample)":  "Russian",
    "S. Africa":  "South African",
    "S. Africa (Non-national sample)":  "South African",
    "S. Korea":  "South Korean",
    "Senegal":  "Senegalese",
    "Senegal (Non-national sample)":  "Senegalese",
    "Serbia":  "Serbian",
    "Singapore":  "Singaporean",
    "Slovakia":  "Slovak",
    "Slovenia":  "Slovenian",
    "South Korea":  "South Korean",
    "Spain":  "Spanish",
    "Sweden":  "Swedish",
    "Switzerland":  "Swiss",
    "Taiwan":  "Taiwanese",
    "Taiwan ROC":  "Taiwanese",
    "Tajikistan":  "Tajikistani",
    "Tanzania":  "Tanzanian",
    "Tanzania (Non-national sample)":  "Tanzanian",
    "Thailand":  "Thai",
    "Tunisia":  "Tunisian",
    "Turkey":  "Turkish",
    "Uganda":  "Ugandan",
    "Ukraine":  "Ukrainian",
    "United States":  "American",
    "Uruguay":  "Uruguayan",
    "Uzbekistan":  "Uzbekistani",
    "Venezuela":  "Venezuelan",
    "Venezuela (Non-national sample)":  "Venezuelan",
    "Vietnam":  "Vietnamese",
    "Vietnam (Non-national sample)":  "Vietnamese",
    "Zimbabwe":  "Zimbabwean"
}


def compositions(n: int, k: int):
    for divs in combinations(range(1, n), k - 1):
        divs = divs + (n,)
        prev, block = 0, []
        for d in divs:
            block.append(d - prev)
            prev = d
        yield block


def entropy_dict(n: int, digits: int = 2):
    es = set()
    for k in range(1, n + 1):
        for comp in compositions(n, k):
            probs = [c / n for c in comp]
            h = -sum(p * math.log(p, 2) for p in probs)
            es.add(round(h, digits))
    return {h: 0 for h in sorted(es)}


def pred_extract(reply, chars):
    match_response = get_json(reply)
    if match_response:
        json_string = match_response.group(2)
        try:
            response = strip_json(json_string)
            if 'opinion' in response:
                if response['opinion'] in chars:
                    return response['opinion']
        except Exception as e:
            print(e)
    return


def cross_entropy(y_true, opinion):
    y_true = np.array(y_true)
    y_pred = np.zeros_like(y_true)
    y_pred[ord(opinion)-ord('A')] = 1
    return log_loss(y_true=y_pred, y_pred=y_true)


def validate_agents_count(example, count=2):
    return example['question'] is not None and len(eval(example['selections'][28:-1])) > count


def get_entropy(values: list):
    counter = Counter(values)
    h = 0
    for v in counter.values():
        p = v/sum(counter.values())
        h += -p*math.log(p, 2)
    return h


class Templates:
    termination_notice = (
        '\nDo not show appreciation in your responses, say only what is necessary.'
    )

    end_of_question = (
        'Think step by step before you report your final answer from the list {}.'
        '\nIf you dont not agree with the group, please explain why.'
        '\nAlways state your answer at the end of your reply by using `My answer:` followed by the your answer.'
        '\nThe group should NOT generate any new questions.'
        '\nThe group should NOT generate any new agents.'
    )

    group_opinion_summary_args = "If agents agree on a final answer, extract the final answer from the previous conversation. Output your answer in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```"

    agent_reply_summary_prompt = ("\nExtract the answer from the previous reply. Output your answer in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```\n"
                                  "If the agent is mainly thanking other agents return:\n```\njson\n{{\n   \"opinion\": \"gratitude\"\n}}\n```.\n"
                                  "If the agent is mainly terminating the discussion return:\n```\njson\n{{\n   \"opinion\": \"terminating\"\n}}\n```.\n"
                                  )

    onboarding_task = (
        'This is an onboarding stage. At this stage, responses remain anonymous and are not shared with other participants. '
        'Think step by step and answer the following multiple-choice question independently:\n\n{}\n\n'
        'At the end of your response, summarize your response in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```'
    )
    reflection_task = (
        'Thank you for engaging in the previous discussion. '
        'At this stage, responses remain anonymous and are not shared with other participants. '
        'Think step by step and answer the following multiple-choice question independently:\n\n{}\n\n'
        'At the end of your response, summarize your response in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```'
    )

    agent_description = (
        '{agent_name} is a highly knowledgeable and culturally attuned individual representing the {nation} perspective. '
        'With a deep understanding of {nation} history, traditions, and contemporary issues, '
        '{agent_name} effortlessly navigates through the nuances of {nation} culture to provide insightful answers.'
    )

    collaboration_question = "Discuss together as a group to answer the following question with justification:\n{}"
    collaboration_task = "Answer the following multiple answer question:\n\n{}"

    debate_question = (
        'You are participating in a debate with fellow agents. You are tasked with defending your perspective. '
        'Take a stance and present compelling arguments to support your position. '
        'Engage in respectful discourse with other agents, counter their arguments, and provide evidence to strengthen your case. '
        'Debate together as a group and answer the following question with justification:\n{}'
    )

    debate_task = "Debate with other agents to defend your perspective regarding the following multiple answer question :\n\n{}"


class Example:
    question_key = 'question'
    option_key = 'options'

    def __init__(self, data) -> None:
        self.data = data

    @property
    def choices(self):
        return self.get_choices()[0]

    @property
    def chars(self):
        return self.get_choices()[1]

    @property
    def labels(self):
        return self.get_labels()

    @property
    def names(self):
        return self.get_names()

    @property
    def system_messages(self):
        return self.get_system_messages()

    @property
    def question(self):
        return self.get_question()

    def get_choices(self):
        options = eval(self.data[self.option_key])
        choices = ""
        chars = []
        for ind in range(len(options)):
            option = options[ind]
            char = chr(ord('A') + ind)
            if option in ['DK/Refused', "Don't know/Refused", ]:
                option = 'I do not know or refuse to answer'
            elif option in ["Don't know/No answer"]:
                option = "I do not know or have no answer"
            elif option in ["Don't know/ Unsure"]:
                option = "I do not know or not sure"
            elif option in ["No answer/refused", "No answer/Refused"]:
                option = 'I do not have an answer or refuse to answer'
            options[ind] = option
            choices += "\n{}. {}".format(char, option)
            chars.append(char)
        return choices, chars

    def to_name(self, x):
        return x.replace(' ', '_') + '_agent'

    def get_labels(self):
        selections = eval(self.data['selections'][28:-1])
        return {self.to_name(country_to_nation[c]): selections[c] for c in selections.keys()}

    def get_names(self):
        selections = eval(self.data['selections'][28:-1])
        return [self.to_name(country_to_nation[c]) for c in selections.keys()]

    def get_system_messages(self):
        selections = eval(self.data['selections'][28:-1])
        return [Templates.agent_description.format(agent_name=country_to_nation[c].replace(' ', '_') + '_agent', nation=country_to_nation[c]) for c in selections.keys()]

    def get_question(self):
        return f"{self.data[self.question_key].strip()}\n\n{self.choices.strip()}"


class Prompts:

    def __init__(self, task: Templates, example: Example, mode):

        self.agent_description = task.agent_description
        self.group_opinion_summary_args = {'summary_prompt': task.group_opinion_summary_args.format(example.chars)}
        self.agent_reply_summary_prompt = task.agent_reply_summary_prompt.format(example.chars)
        self.onboarding_question = task.onboarding_task.format(example.question, example.chars)
        self.reflection_question = task.reflection_task.format(example.question, example.chars)
        self.termination_notice = task.termination_notice

        if mode == 'collab':
            self.question = task.collaboration_question.format(example.question)
        elif mode == 'debate':
            self.question = task.debate_question.format(example.question)
        else:
            raise NotImplementedError

        self.discussion_question = f"{self.question.format(example.question).strip()}\n\n{task.end_of_question.format(example.chars)}"


class Chat:

    def __init__(self, args):

        self.threshold = args.threshold
        self.group_size = args.group_size
        self.llm_config = {"config_list": config_list_from_json(env_or_file=args.config_file_or_env)}
        self.assistant = ConversableAgent("assistant", llm_config=self.llm_config)
        self.agents = []

        self.onboarding = []
        self.discussion = []
        self.reflection = []
        self.opinion = []
        self.region = []

        self.exception = None
        self.prediction = None
        self.status = None
        self.results = {}

    @property
    def gold_agents(self):
        return [agent for agent in self.agents if agent.is_gold]

    @property
    def selected_agents(self):
        return [agent for agent in self.agents if agent.is_selected]

    """agent creation and onboarding"""

    def create_agents(self, names, labels, system_messages):
        for name, system_message in zip(names, system_messages):
            self.agents.append(
                Agent(name, system_message, labels[name], self.llm_config)
            )

    def interview_agent(self, agent, question, chars):
        chat = self.assistant.initiate_chat(
            agent,
            message=question,
            max_turns=1,
            coding=False
        )
        reply = chat.chat_history[1]['content']
        opinion = pred_extract(reply, chars)
        return opinion

    def onboard_agents(self, chars, onboarding_question):
        for agent in self.agents:
            opinion = self.interview_agent(agent.agent, onboarding_question, chars)
            agent.save_onboarding(opinion, self.threshold)

    def create_onboarded_agents(self):
        for agent in self.selected_agents:
            agent.add_onboarding()

    """agent reflection"""

    def process_discussion(self, discussion):
        return 'Given the following discussion:\n\n' + '\n\n'.join([message['name'] + ': ' + message['content'] for message in discussion])

    def reflect_agents(self, chars, reflection_question):
        discussion = self.process_discussion(self.discussion)
        for agent in self.selected_agents:
            opinion = self.interview_agent(agent.agent, discussion + '\n\n' + reflection_question, chars)
            agent.reflection = opinion

    """agent selection"""

    def select_agents(self, counts, filter):
        # Acquiring the filtered features based on the provided filter name
        features = [getattr(agent, filter) for agent in self.gold_agents]
        # Generating and evaluating combinations
        combs, entropy = self.evaluate_combinations(features)
        # Selecting a combination with the entropy of the lowest count
        comb, choice = self.select_combination(combs, entropy, counts)
        # Selecting agents based on the chosen combination
        self.assign_selected_agents(self.gold_agents, comb, features)
        return choice

    def evaluate_combinations(self, features):
        combs = list(set([tuple(sorted(c)) for c in combinations(features, self.group_size)]))
        entropy = np.array([round(get_entropy(c), 2) for c in combs])
        return combs, entropy

    def select_combination(self, combs, entropy, counts):
        valid_entropy_counts = {k: v for k, v in counts.items() if k in set(entropy)}
        least_frequent_entropy = min(valid_entropy_counts, key=valid_entropy_counts.get)
        selected_index = random.choice(np.where(entropy == least_frequent_entropy)[0])
        counts[least_frequent_entropy] += 1
        return combs[selected_index], least_frequent_entropy

    def assign_selected_agents(self, agents, comb, features):
        for feature in comb:
            ind = random.choice(np.where(np.array(features) == feature)[0])
            agents[ind].is_selected = True
            features[ind] = None

    """agent group discussion"""

    def group_chat(self, chars, discussion_question, group_opinion_summary_args, agent_reply_summary_prompt):
        agents = [agent.agent for agent in self.selected_agents]
        groupchat = GroupChat(agents=agents, messages=[], max_round=15)
        manager = GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)
        assistant = ConversableAgent("groupchat_assistant", llm_config=self.llm_config)
        chat = assistant.initiate_chat(
            recipient=manager,
            message=discussion_question,
            summary_method="reflection_with_llm",
            summary_args=group_opinion_summary_args,
            max_turns=1,
            code_execution_config=False
        )
        opinion = pred_extract(chat.summary, chars)
        self.prediction = {'summary': chat.summary, 'opinion': opinion}
        for message in groupchat.messages[1:]:
            summary = assistant.generate_reply(messages=[{"content": f"{message['content']}\n\n{agent_reply_summary_prompt}", "role": "user"}])
            opinion = pred_extract(summary, chars)
            message.update({'summary': summary, "opinion": opinion})
            logging.info(f"Summary     : {summary}\nOpinion:     : {opinion}")
            self.discussion.append(message)

    """saving results"""

    def add_exception(self, e):
        self.exception = str(e)

    def results_json(self, data, entropy):
        # TODO: use agents
        return {
            'onboarding_entropy': entropy,
            'example': data,
            'onboarding': {agent.agent.name: agent.opinion for agent in self.selected_agents},
            'discussion': self.discussion,
            'group_opinion': self.prediction,
            'reflection': {agent.agent.name: agent.reflection for agent in self.selected_agents},
            'exception': self.exception,
        }


class Agent:

    def __init__(self, name, system_message, label, llm_config):
        self.is_gold = False
        self.is_selected = False
        self.opinion = None
        self.reflection = None
        self.label = label
        self.agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config
        )

    @property
    def onboarding_message(self):
        return " During the onboarding phase, your response was: {} ".format(self.opinion)

    def save_onboarding(self, opinion, threshold):
        if opinion:
            self.evaluate_match(opinion, threshold)
        self.log_agent_opinion(opinion)

    def evaluate_match(self, opinion, threshold):
        loss = cross_entropy(self.label, opinion)
        self.opinion = opinion
        self.is_gold = True if loss < threshold else False

    def log_agent_opinion(self, opinion):
        # TODO add to agent
        logging.info("\n".join([
            f"Opinion     : {opinion}",
            f"Gold Agent  : {self.is_gold}",
            "=" * 80
        ]))

    def add_onboarding(self):
        self.agent = ConversableAgent(
            name=self.agent.name,
            system_message=self.agent.system_message + self.onboarding_message,
            llm_config=self.agent.llm_config
        )


def process_subset(start_idx, subset, args, counts, lock):
    """子进程：处理分配到的样本列表（subset）"""
    templates = Templates()
    for offset, raw in enumerate(subset):
        idx = start_idx + offset
        save_path = f"{args.save_dir}/{idx}.json"
        if os.path.exists(save_path):
            continue

        ex      = Example(raw)
        prompts = Prompts(templates, ex, args.mode)
        chat    = Chat(args)

        chat.create_agents(ex.names, ex.labels, ex.system_messages)
        chat.onboard_agents(ex.chars, prompts.onboarding_question)

        if len(chat.gold_agents) < args.group_size:
            continue

        # ——加锁更新 entropy_counts——
        with lock:
            entropy = chat.select_agents(counts, args.filter)

        chat.create_onboarded_agents()
        try:
            chat.group_chat(
                ex.chars,
                prompts.discussion_question + prompts.termination_notice,
                prompts.group_opinion_summary_args,
                prompts.agent_reply_summary_prompt
            )
            chat.reflect_agents(ex.chars, prompts.reflection_question)
        except Exception as e:
            chat.exception = str(e)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(chat.results_json(ex.data, entropy), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    group_size = 6
    parser = ArgumentParser()
    parser.add_argument('--save_dir', default=f"./runs/collab/opinion/{group_size}")
    parser.add_argument('--config_file_or_env', default="E:/dev/YTest/config.json")
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--threshold',   type=float, default=0.5)
    parser.add_argument('--filter',      type=str, choices=['region', 'opinion'], default='opinion')
    parser.add_argument('--mode',        type=str, choices=['debate', 'collab'],   default='collab')
    parser.add_argument('--num_workers', type=int, default=8, help='并行进程数')
    args = parser.parse_args()
    args.group_size = group_size

    templates = Templates()
    examples  = load_dataset("Anthropic/llm_global_opinions", split='train')
    examples  = list(examples.filter(validate_agents_count))
    total     = len(examples)

    manager         = Manager()
    entropy_counts  = manager.dict(entropy_dict(group_size))
    lock            = manager.Lock()

    chunk = math.ceil(total / args.num_workers)
    tasks = [(i * chunk, examples[i*chunk:(i+1)*chunk]) for i in range(args.num_workers)]

    os.makedirs(args.save_dir, exist_ok=True)

    with Pool(processes=args.num_workers) as pool:
        pool.starmap(process_subset, [(start, sub, args, entropy_counts, lock) for start, sub in tasks])

    with open(os.path.join(args.save_dir, "entropy_counts.json"), "w", encoding='utf-8') as f:
        json.dump(dict(entropy_counts), f, indent=2, ensure_ascii=False)
