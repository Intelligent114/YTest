import os
import sys
import warnings
import re
import math
import json
import pandas as pd
import numpy as np
import mpld3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
from matplotlib import MatplotlibDeprecationWarning, cm
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Discussion:

    def __init__(self, id):
        self.id = id
        self.discussion = []
        self.agents = []
        self.regions = []
        self.onboarding_opinions = defaultdict(str)
        self.reflection_opinions = defaultdict(str)
        self.intermediate_opinions = defaultdict(list)
        self.intermediate_replies = defaultdict(list)
        self.intermediate_opinions_with_order = []
        self.gratitude = defaultdict(int)
        self.terminate = defaultdict(int)
        self.analyzing = defaultdict(int)
        self.first_opinion = None
        self.group_opinion = None
        self.num_agents = 0

    def get_freq(self, x):
        counter = Counter(self.onboarding_opinions.values())
        return counter[x] / sum(counter.values())

    def get_num_replies(self):
        return sum([len(v) for v in self.intermediate_opinions.values()])

    @staticmethod
    def _get_entropy(values):
        counter = Counter(values)
        h = 0
        for v in counter.values():
            p = v / sum(counter.values())
            h += -p * math.log(p, 2)
        return h

    def get_entropy(self, step):
        if step == 'onboarding_opinion':
            values = self.onboarding_opinions.values()
        elif step == 'reflection_opinion':
            values = self.reflection_opinions.values()
        elif step == 'intermediate_opinion':
            values = [v for vs in self.intermediate_opinions.values() for v in vs]
        elif step == 'region':
            values = self.regions
        else:
            raise NotImplementedError

        return self._get_entropy(values)

    def ratio_reflection_diff_from_onboarding(self):
        return sum([1 if (self.reflection_opinions[c] != ob and self.reflection_opinions[c] is not None) else 0 for c, ob in self.onboarding_opinions.items()]) / self.num_agents

    def ratio_intermediate_diff_from_onboarding(self):
        res = []
        for c, opinions in self.intermediate_opinions.items():
            res.append(sum([1 if self.onboarding_opinions[c] != a else 0 for a in opinions]) / len(opinions))
        return sum(res) / self.num_agents

    def onboarding_compared_to_reflection(self):
        res = []
        for c, ob in self.onboarding_opinions.items():
            rf = self.reflection_opinions[c]
            if ob == None or rf == None:
                change_onboarding_reflection = 'N/A'
            elif rf != ob:
                change_onboarding_reflection = 1
            else:
                change_onboarding_reflection = 0

            if ob == None or self.intermediate_opinions[c] == []:
                ratio_intermediate_equal_onboarding = 'N/A'
                ratio_intermediate_equal_reflection = 'N/A'
            else:
                ratio_intermediate_equal_onboarding = self.intermediate_opinions[c].count(ob) / len(self.intermediate_opinions[c])
                ratio_intermediate_equal_reflection = self.intermediate_opinions[c].count(rf) / len(self.intermediate_opinions[c])

            res.append((change_onboarding_reflection, self.get_freq(ob), ratio_intermediate_equal_onboarding, ratio_intermediate_equal_reflection))
        return res

    def group_opinion_compared_to_reflection(self):
        res = []
        for c, ob in self.onboarding_opinions.items():
            rf = self.reflection_opinions[c]
            if ob == None or rf == None:
                change_onboarding_reflection = 'N/A'
            elif rf != ob:
                change_onboarding_reflection = 1
            else:
                change_onboarding_reflection = 0

            if self.group_opinion == None or self.intermediate_opinions[c] == []:
                ratio_intermediate_equal_group_opinion = 'N/A'
                ref_equal_group_opinion = 'N/A'
            else:
                ratio_intermediate_equal_group_opinion = self.intermediate_opinions[c].count(self.group_opinion) / len(self.intermediate_opinions[c])
                if self.group_opinion == rf:
                    ref_equal_group_opinion = 1
                else:
                    ref_equal_group_opinion = 0

            res.append((change_onboarding_reflection, self.get_freq(ob), ref_equal_group_opinion, ratio_intermediate_equal_group_opinion))
        return res

    def group_opinion_same_as_initiator(self):
        return self.group_opinion == list(self.discussion[0].values())[0]

    def initiator_changes_opinion(self):
        initiator, discussion_opinion = list(self.discussion[0].items())[0]
        opinion = self.onboarding_opinions[initiator]
        return True if discussion_opinion != opinion else False

    def initiator_opinion_prob(self):
        initiator, discussion_opinion = list(self.discussion[0].items())[0]
        opinion = self.onboarding_opinions[initiator]
        return self.get_freq(opinion), self.get_freq(discussion_opinion)

    def detect_impersonation(self):
        false_agent_count = 0
        for name, replies in self.intermediate_replies.items():
            name = name.replace('_', ' ')[:-6]
            for reply in replies:
                matches = re.findall("As an? ([\w\s]+) agent", reply)
                for match in matches:
                    if match != name:
                        print(self.id)
                        false_agent_count += 1
                        print(name, '->', match)
                        pass
                    # print(f'[{self.intermediate_replies[ind-1][0]}]', self.intermediate_replies[ind-1][1])
                    # print('-'*80)
                    # print(f'[{self.intermediate_replies[ind][0]}]', self.intermediate_replies[ind][1])
                    # print('*'*80)

        return false_agent_count

    def detect_confabulation(self):
        unseen_opinions = 0
        intermediate_opinions = [a for ans in self.intermediate_opinions.values() for a in ans]
        previous_ans = set(intermediate_opinions + list(self.onboarding_opinions.values()))
        for ans in self.reflection_opinions.values():
            if ans and ans not in previous_ans:  # and ans in self.chars:
                print(ans, previous_ans)
                unseen_opinions += 1
        return unseen_opinions


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



def get_ratio_over_bin(df, x, col, total='size'):
    return x[total] / df[df[col] == x[col]][total].sum()



def main():
    mode = 'collab'
    filter = "opinion"
    scale = 3

    num_groups = 6

    assert mode in ['collab', 'debate']

    save_dir = f"runs/{mode}/{filter}/{num_groups}/"
    results_dir = f"results/{mode}/{filter}/{num_groups}/"
    os.makedirs(results_dir, exist_ok=True)
    results_dir = results_dir + f"{mode}-{filter}-"

    exceptions = 0
    no_group_opinion = 0
    no_reflection_opinion = 0
    no_onboarding_opinion = 0

    discussions = []

    examples = os.listdir(os.path.join(save_dir))
    for e in sorted(examples):
        try:
            with open(os.path.join(save_dir, e), 'r', encoding='utf-8') as f:
                j = json.load(f)
        except Exception as ex:
            print(ex)
            pass

        agents = j['onboarding'].keys()

        if j['group_opinion'] is None:
            exceptions += 1
            print(os.path.join(save_dir, e), len(agents))
            os.remove(os.path.join(save_dir, e))
            continue

        example = Example(j['example'])
        group_opinion = j['group_opinion']['opinion']
        onboarding = j['onboarding']
        reflection = j['reflection']

        if j['exception']:
            exceptions += 1
            print(os.path.join(save_dir, e), len(agents))
            os.remove(os.path.join(save_dir, e))

        if not group_opinion:
            no_group_opinion += 1
            continue
        else:
            discussion = Discussion(e[:-num_groups])
            discussion.group_opinion = group_opinion
            discussion.agents = agents
            discussion.num_agents = len(agents)
            if len(agents) != num_groups:
                print(os.path.join(save_dir, e), len(agents))
                os.remove(os.path.join(save_dir, e))

            discussion.onboarding_opinions = onboarding

            d = j['discussion']

            for ind, m in enumerate(d):
                name = m["name"]
                opinion = m["opinion"]
                content = m["content"]
                discussion.discussion.append({name: opinion})
                if opinion:
                    discussion.intermediate_opinions[name].append(opinion)
                    discussion.intermediate_opinions_with_order.append((name, opinion))
                    discussion.intermediate_replies[name].append(content)

            discussion.reflection_opinions = reflection
            discussions.append(discussion)

    print('Exceptions', exceptions)
    print('Number of examples', len(discussions))

    filter = 'onboarding_opinion' if filter == 'opinion' else 'region'
    onboarding_entropy = np.array([d.get_entropy(filter) for d in discussions])  # also in examples
    intermediate_ents = np.array([d.get_entropy('intermediate_opinion') for d in discussions])
    reflection_ents = np.array([d.get_entropy('reflection_opinion') for d in discussions])
    region_ents = np.array([d.get_entropy('region') for d in discussions])

    num_replies = [d.get_num_replies() for d in discussions]
    group_opinion_freq = np.array([d.get_freq(d.group_opinion) for d in discussions])
    group_opinion_same_as_initiator = [d.group_opinion_same_as_initiator() for d in discussions]
    initiator_changes_opinion = [d.initiator_changes_opinion() for d in discussions]
    initiator_opinion_prob = [d.initiator_opinion_prob() for d in discussions]

    ratio_reflection_diff_from_onboarding = np.array([d.ratio_reflection_diff_from_onboarding() for d in discussions])
    ratio_intermediate_diff_from_onboarding = np.array([d.ratio_intermediate_diff_from_onboarding() for d in discussions])
    onboarding_compared_to_reflection = [d.onboarding_compared_to_reflection() for d in discussions]
    group_opinion_compared_to_reflection = [d.group_opinion_compared_to_reflection() for d in discussions]

    false_agent = [d.detect_impersonation() for d in discussions]
    unseen_answers = [d.detect_confabulation() for d in discussions]

    false_agent_ratio = [i / n for i, n in zip(false_agent, num_replies) if n]
    print("False agent percentage:", round(sum(false_agent_ratio) / len(false_agent_ratio) * 100, 2), '%')
    print("Unseen answers percentage:", round(sum(unseen_answers) / (num_groups * len(unseen_answers)) * 100, 2), '%')

    # ##### Figure 4:  Initiator Changes Opinion

    df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'initiator_changes_opinion': initiator_changes_opinion, })
    df['onboarding_entropy'] = df['onboarding_entropy'].round(2).astype(str)
    df = df.groupby(['onboarding_entropy', 'initiator_changes_opinion'], as_index=False, dropna=False).size()
    df['ratio'] = df.apply(lambda x: get_ratio_over_bin(df, x, 'onboarding_entropy'), axis=1)
    df = df.sort_values(['onboarding_entropy', 'initiator_changes_opinion'], ascending=[True, False])
    entropy_categories = df['onboarding_entropy'].unique().tolist()
    df['onboarding_entropy'] = pd.Categorical(df['onboarding_entropy'], categories=entropy_categories, ordered=True)
    label_map = {True: r'$I \neq O$', False: r'$I = O$'}
    df['initiator_changes_opinion_label'] = df['initiator_changes_opinion'].map(label_map)

    # Prepare data for stacked bars
    df_pivot = df.pivot_table(index='onboarding_entropy', columns='initiator_changes_opinion_label', values='ratio', aggfunc='sum', fill_value=0)

    # Plot
    plt.figure(figsize=(8, 6))
    ax = df_pivot.plot(kind='bar', stacked=True, edgecolor='black', width=0.8)

    # Customizing the chart
    ax.set_xlabel("Entropy", fontsize=14, family="Times New Roman")
    ax.set_ylabel("Ratio", fontsize=14, family="Times New Roman")
    plt.xticks(fontsize=12, family="Times New Roman")
    plt.yticks(fontsize=12, family="Times New Roman")
    plt.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=12)
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))

    # Saving the plot
    html_filename = f"{results_dir}initiator_changes_opinion.html"
    mpld3.save_html(plt.gcf(), html_filename)
    plt.savefig(f"{results_dir}initiator_changes_opinion.png", dpi=300*scale, bbox_inches='tight')
    # plt.show()

    # ##### Figure 3: Initiators Dominate Group Prediction

    # Assuming df and other variables are already defined as in your code
    df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'group_opinion_same_as_initiator': group_opinion_same_as_initiator})
    df['onboarding_entropy'] = df['onboarding_entropy'].round(2).astype(str)
    df = df.groupby(['onboarding_entropy', 'group_opinion_same_as_initiator'], as_index=False, dropna=False).size()
    df['ratio'] = df.apply(lambda x: get_ratio_over_bin(df, x, 'onboarding_entropy'), axis=1)
    df = df[df['size'] != 0]
    df = df.sort_values(['onboarding_entropy', 'group_opinion_same_as_initiator', 'ratio'], ascending=[True, False, True])
    unique_entropy = df['onboarding_entropy'].unique().tolist()
    df['onboarding_entropy'] = pd.Categorical(df['onboarding_entropy'], categories=unique_entropy, ordered=True)
    label_map = {True: r'$G = I$', False: r'$G \neq I$'}
    df['group_opinion_label'] = df['group_opinion_same_as_initiator'].map(label_map)

    # Prepare data for stacked bars
    df_pivot = df.pivot_table(index='onboarding_entropy', columns='group_opinion_label', values='ratio', aggfunc='sum', fill_value=0)

    # Plot
    plt.figure(figsize=(8, 6))
    ax = df_pivot.plot(kind='bar', stacked=True, edgecolor='black', width=0.8)

    # Customizing the chart
    ax.set_xlabel("Entropy", fontsize=16, family="Times New Roman")
    ax.set_ylabel("Ratio", fontsize=16, family="Times New Roman")
    plt.xticks(fontsize=14, family="Times New Roman")
    plt.yticks(fontsize=14, family="Times New Roman")
    plt.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)
    plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.90))

    # Saving the plot
    plt.savefig(f"{results_dir}group_opinion_same_as_initiator.png", dpi=300*scale, bbox_inches='tight')
    mpld3.save_html(plt.gcf(), f"{results_dir}group_opinion_same_as_initiator.html")
    # mpld3.display()
    # plt.show()

    # ##### Figure 2: Group Opinion follows the Distribution of Opinions during onboarding

    # Assuming df and other variables are already defined as in your code
    df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'group_opinion_freq': group_opinion_freq})

    df['onboarding_entropy'] = df['onboarding_entropy'].round(2).astype(str)
    df = df.groupby(['onboarding_entropy', 'group_opinion_freq'], as_index=False, dropna=False).size()
    df['ratio'] = df.apply(lambda x: get_ratio_over_bin(df, x, 'onboarding_entropy'), axis=1)
    df = df[df['size'] != 0]
    df = df.sort_values(['onboarding_entropy', 'group_opinion_freq', 'ratio'])
    entropy_categories = df['onboarding_entropy'].unique().tolist()
    df['onboarding_entropy'] = pd.Categorical(df['onboarding_entropy'], categories=entropy_categories, ordered=True)

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('lightgray')
    ax.set_facecolor('lightgray')

    # Color map setup
    cmap = cm.get_cmap('bwr')
    norm = mcolors.Normalize(vmin=df['group_opinion_freq'].min(), vmax=df['group_opinion_freq'].max())

    # Calculate the stacked bars
    current_x = 0
    bar_width = 0.8  # Bar width
    x_positions = []
    cumulative_ratios = {}

    # Loop through each onboarding_entropy and plot the stacked bars for each group_opinion_freq
    for entropy in df['onboarding_entropy'].unique():
        subset = df[df['onboarding_entropy'] == entropy]

        # Calculate cumulative heights for stacked bars
        cumulative_height = 0
        for idx, row in subset.iterrows():
            ratio = row['ratio']
            x_position = current_x

            # Plotting the stacked bar part for each group_opinion_freq
            color = cmap(norm(row['group_opinion_freq']))
            ax.bar(x_position, ratio, bottom=cumulative_height, color=color, width=bar_width)

            cumulative_height += ratio  # Update cumulative height

        x_positions.append(current_x)  # Store the x position for this onboarding_entropy
        current_x += 1  # Increment the x position for next onboarding_entropy

    # Set the X-ticks to the middle of each stacked bar group
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['onboarding_entropy'].unique(), rotation=45, ha='right', fontsize=12, family="Times New Roman")

    # Labels and title
    ax.set_xlabel("Entropy", fontsize=14, family="Times New Roman")
    ax.set_ylabel("Ratio", fontsize=14, family="Times New Roman")

    # Color bar setup
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(df['group_opinion_freq'])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Probability", fontsize=14, family="Times New Roman")

    # Tight layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f"{results_dir}onboarding_probability_by_entropy.png", dpi=300 * scale, bbox_inches='tight')
    mpld3.save_html(fig, f"{results_dir}onboarding_probability_by_entropy.html")
    # mpld3.display()
    # plt.show()

    # ##### Tables:

    df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'onboarding_compared_to_reflection': onboarding_compared_to_reflection})
    df['onboarding_entropy'] = df['onboarding_entropy'].round(2)
    df = df.explode('onboarding_compared_to_reflection')
    df[['change_onboarding_reflection', 'prob_onboarding', 'ratio_intermediate_equal_onboarding', 'ratio_intermediate_equal_reflection']] = pd.DataFrame(df.onboarding_compared_to_reflection.tolist(),
                                                                                                                                                         index=df.index)

    df.reset_index(inplace=True)
    df = df.drop(columns=['onboarding_compared_to_reflection', 'index'])
    df['Count'] = 1
    df = df.groupby(['onboarding_entropy', 'prob_onboarding', 'change_onboarding_reflection', 'ratio_intermediate_equal_onboarding', 'ratio_intermediate_equal_reflection']).Count.count().reset_index()
    df = df[df['change_onboarding_reflection'] != 'N/A']
    df = df[df['ratio_intermediate_equal_onboarding'] != 'N/A']

    print('Peer Pressure', mode)

    for ind, ((entropy, prob_onboarding), group) in enumerate(df.groupby(['onboarding_entropy', 'prob_onboarding', ])):
        for change_onboarding_reflection, _group in group.groupby(['change_onboarding_reflection']):
            sum_ratio_intermediate_equal_reflection = group['ratio_intermediate_equal_reflection'].map(float) * _group['Count']
            average_ratio_intermediate_nequal_reflection = sum_ratio_intermediate_equal_reflection.sum() / _group['Count'].sum()
            ratio = _group['Count'].sum() / group['Count'].sum()
            if not change_onboarding_reflection[0]:
                print(",".join(map(str, [entropy, prob_onboarding, round(100 * ratio, 2), round(average_ratio_intermediate_nequal_reflection, 2)])), end='')
            else:
                print(',' + ",".join(map(str, [round(100 * ratio, 2), round(average_ratio_intermediate_nequal_reflection, 2)])))

    df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'group_opinion_compared_to_reflection': group_opinion_compared_to_reflection})
    df['onboarding_entropy'] = df['onboarding_entropy'].round(2)
    df = df.explode('group_opinion_compared_to_reflection')
    df[['change_onboarding_reflection', 'prob_onboarding', 'ref_equal_group_opinion', 'ratio_intermediate_equal_group_opinion']] = pd.DataFrame(df.group_opinion_compared_to_reflection.tolist(),
                                                                                                                                                index=df.index)
    df.reset_index(inplace=True)
    df = df.drop(columns=['group_opinion_compared_to_reflection', 'index'])
    df['Count'] = 1
    df = df.groupby(['onboarding_entropy', 'prob_onboarding', 'change_onboarding_reflection', 'ref_equal_group_opinion', 'ratio_intermediate_equal_group_opinion']).Count.count().reset_index()
    df = df[df['change_onboarding_reflection'] != 'N/A']
    df = df[df['ratio_intermediate_equal_group_opinion'] != 'N/A']

    print('Peer Influence', mode)

    for ind, ((entropy, prob_onboarding), group) in enumerate(df.groupby(['onboarding_entropy', 'prob_onboarding', ])):
        for change_onboarding_reflection, group1 in group.groupby(['change_onboarding_reflection']):
            for ref_equal_group_opinion, group2 in group1.groupby(['ref_equal_group_opinion']):
                sum_ratio_intermediate_equal_group_opinion = group2['ratio_intermediate_equal_group_opinion'].map(float) * group2['Count']
                average_ratio_intermediate_equal_group_opinion = sum_ratio_intermediate_equal_group_opinion.sum() / group1['Count'].sum()
                ratio1 = group1['Count'].sum() / group['Count'].sum()
                ratio2 = group2['Count'].sum() / group1['Count'].sum()
                if not change_onboarding_reflection[0]:
                    if not ref_equal_group_opinion[0]:
                        print(",".join(map(str, [entropy, prob_onboarding, round(100 * ratio1, 2), round(100 * ratio2, 2)])), end=',')
                    else:
                        print(f'{round(100 * ratio2, 2)}', end=',')
                else:
                    if not ref_equal_group_opinion[0]:
                        print(",".join(map(str, [round(100 * ratio1, 2), round(100 * ratio2, 2)])), end=',')
                    else:
                        print(f'{round(100 * ratio2, 2)}', end='\n')


if __name__ == "__main__":
    main()
