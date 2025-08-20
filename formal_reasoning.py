import copy
import random

import numpy as np


class Task:

    def __init__(self, sub, obj, copula, truth, eb, rl=None):
        # statement
        self.sub = sub
        self.obj = obj
        self.copula = copula

        # truth
        self.truth = truth
        self.f = truth.f
        self.c = truth.c

        # evidential_base
        self.eb = eb

        # rules used to derive this task
        self.r = rl

    def string(self):
        return f"<{self.sub}{self.copula}{self.obj}>. %{round(self.f, 3)}; {round(self.c, 3)}% {self.eb}"

    def to_json(self):
        if self.r is None:
            return {"s": self.sub,
                    "o": self.obj,
                    "cp": self.copula,
                    "f": round(self.f, 3),
                    "c": round(self.c, 3),
                    "eb": sorted(list(self.eb))}
        else:
            return {"s": self.sub,
                    "o": self.obj,
                    "cp": self.copula,
                    "f": round(self.f, 3),
                    "c": round(self.c, 3),
                    "eb": sorted(list(self.eb)),
                    "r": self.r}


class Truth:

    def __init__(self, f=1., c=0.9):
        self.f = f
        self.c = c

    @property
    def w(self):
        return self.c * (1 - self.c)

    @property
    def wp(self):
        return self.w + self.f

    @property
    def wn(self):
        return self.w * (1 - self.f)


class TruthFunctions:

    def __init__(self):
        self.tf = {"ded": self.ded,
                   "ded_p": self.ded_p,
                   "ana": self.ana,
                   "ana_p": self.ana_p,
                   "res": self.res,
                   "res_p": self.res_p,
                   "abd": self.abd,
                   "abd_p": self.abd_p,
                   "ind": self.ind,
                   "ind_p": self.ind_p,
                   "exe": self.exe,
                   "exe_p": self.exe_p,
                   "com": self.com,
                   "com_p": self.com_p
                   }

    @staticmethod
    def AND(*values):
        return np.prod(values)

    @staticmethod
    def OR(*values):
        return 1 - np.prod([1 - x for x in values])

    def ded(self, truth_1: Truth, truth_2: Truth):
        return Truth(self.AND(truth_1.f, truth_2.f), self.AND(truth_1.f, truth_2.f, truth_1.c, truth_2.c))

    def ded_p(self, truth_1: Truth, truth_2: Truth):
        return self.ded(truth_2, truth_1)

    def ana(self, truth_1: Truth, truth_2: Truth):
        return Truth(self.AND(truth_1.f, truth_2.f), self.AND(truth_2.f, truth_1.c, truth_2.c))

    def ana_p(self, truth_1: Truth, truth_2: Truth):
        return self.ana(truth_2, truth_1)

    def res(self, truth_1: Truth, truth_2: Truth):
        return Truth(self.AND(truth_1.f, truth_2.f), self.AND(self.OR(truth_1.f, truth_2.f), truth_1.c, truth_2.c))

    def res_p(self, truth_1: Truth, truth_2: Truth):
        return self.res(truth_2, truth_1)

    def abd(self, truth_1: Truth, truth_2: Truth):
        wp = self.AND(truth_1.f, truth_2.f, truth_1.c, truth_2.c)
        w = self.AND(truth_1.f, truth_1.c, truth_2.c)
        return Truth(wp / w, w / (w + 1))

    def abd_p(self, truth_1: Truth, truth_2: Truth):
        return self.abd(truth_2, truth_1)

    def ind(self, truth_1: Truth, truth_2: Truth):
        wp = self.AND(truth_1.f, truth_2.f, truth_1.c, truth_2.c)
        w = self.AND(truth_2.f, truth_1.c, truth_2.c)
        return Truth(wp / w, w / (w + 1))

    def ind_p(self, truth_1: Truth, truth_2: Truth):
        return self.ind(truth_2, truth_1)

    def exe(self, truth_1: Truth, truth_2: Truth):
        wp = self.AND(truth_1.f, truth_2.f, truth_1.c, truth_2.c)
        w = self.AND(truth_1.f, truth_2.f, truth_1.c, truth_2.c)
        return Truth(wp / w, w / (w + 1))

    def exe_p(self, truth_1: Truth, truth_2: Truth):
        return self.exe(truth_2, truth_1)

    def com(self, truth_1: Truth, truth_2: Truth):
        wp = self.AND(truth_1.f, truth_2.f, truth_1.c, truth_2.c)
        w = self.AND(self.OR(truth_1.f, truth_2.f), truth_1.c, truth_2.c)
        return Truth(wp / w, w / (w + 1))

    def com_p(self, truth_1: Truth, truth_2: Truth):
        return self.com(truth_2, truth_1)


TFS = TruthFunctions()


def reasoning(task_1: Task, task_2: Task):
    ret = []

    if task_1.sub == task_2.obj and task_1.copula == task_2.copula == "-->":
        # MP, SM -> SP(ded), PS('exe)
        t = Task(task_2.sub, task_1.obj, "-->", TFS.ded(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ded")
        ret.append(t)
        t = Task(task_1.obj, task_2.sub, "-->", TFS.exe_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "exe_p")
        ret.append(t)
        flag = "MP, SM"
    elif task_1.obj == task_2.obj and task_1.copula == task_2.copula == "-->":
        # PM, SM -> SP(abd), PS('abd), S<>P('com)
        t = Task(task_2.sub, task_1.sub, "-->", TFS.abd(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "abd")
        ret.append(t)
        t = Task(task_1.sub, task_2.sub, "-->", TFS.abd_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "abd_p")
        ret.append(t)
        t = Task(task_2.sub, task_1.sub, "<->", TFS.com_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "com_p")
        ret.append(t)
        # t = Task(task_1.sub, task_2.sub, "<->", TFS.com_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com_p")
        # ret.append(t)
        flag = "PM, SM"
    elif task_1.sub == task_2.obj and task_1.copula == "<->" and task_2.copula == "-->":
        # M<>P, SM -> SP('ana)
        t = Task(task_2.sub, task_1.obj, "-->", TFS.ana_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "ana_p")
        ret.append(t)
        flag = "M<>P, SM"
    elif task_1.sub == task_2.sub and task_1.copula == task_2.copula == "-->":
        # MP, MS -> SP(ind), PS('ind), S<>P(com)
        t = Task(task_2.obj, task_1.obj, "-->", TFS.ind(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ind")
        ret.append(t)
        t = Task(task_1.obj, task_2.obj, "-->", TFS.ind_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "ind_p")
        ret.append(t)
        t = Task(task_2.obj, task_1.obj, "<->", TFS.com(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com")
        ret.append(t)
        # t = Task(task_1.obj, task_2.obj, "<->", TFS.com(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com")
        # ret.append(t)
        flag = "MP, MS"
    elif task_1.obj == task_2.sub and task_1.copula == task_2.copula == "-->":
        # PM, MS -> SP(exe), PS('ded)
        t = Task(task_2.obj, task_1.sub, "-->", TFS.exe(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "exe")
        ret.append(t)
        t = Task(task_1.sub, task_2.obj, "-->", TFS.ded_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "ded_p")
        ret.append(t)
        flag = "PM, MS"
    elif task_1.sub == task_2.sub and task_1.copula == "<->" and task_2.copula == "-->":
        # M<>P, MS -> PS('ana)
        t = Task(task_1.obj, task_2.obj, "-->", TFS.ana_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb),
                 "ana_p")
        ret.append(t)
        flag = "M<>P, MS"
    elif task_1.sub == task_2.obj and task_1.copula == "-->" and task_2.copula == "<->":
        # MP, S<>M -> SP(ana)
        t = Task(task_2.sub, task_1.obj, "-->", TFS.ana(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ana")
        ret.append(t)
        flag = "MP, S<>M"
    elif task_1.obj == task_2.obj and task_1.copula == "-->" and task_2.copula == "<->":
        # PM, S<>M -> PS(ana)
        t = Task(task_1.sub, task_2.sub, "-->", TFS.ana(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ana")
        ret.append(t)
        flag = "PM, S<>M"
    elif task_1.sub == task_2.obj and task_1.copula == task_2.copula == "<->":
        # M<>P, S<>M -> S<>P(res)
        t = Task(task_2.sub, task_1.obj, "<->", TFS.res(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "res")
        ret.append(t)
        # t = Task(task_1.obj, task_2.sub, "<->", TFS.res(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "res")
        # ret.append(t)
        flag = "M<>P, S<>M"

    return ret


_truth_categories = [
    (0.0, 0.2, ["is false", "completely false", "does not hold", "has been refuted", "is incorrect"]),
    (0.2, 0.4, ["is mostly false", "tends to be false", "seems incorrect", "barely holds", "is generally wrong"]),
    (0.4, 0.6,
     ["is unknown", "is undetermined", "cannot be classified", "its truth is unclear", "is neither true nor false"]),
    (0.6, 0.8, ["is mostly true", "tends to be true", "seems correct", "largely holds", "is generally valid"]),
    (0.8, 1.01, ["is true", "completely holds", "has been confirmed", "is correct", "is a fact"])
]

_inheritance_templates = [
    "{sub} is a type of {obj}",
    "Every {sub} is an instance of {obj}",
    "{sub} falls under the category of {obj}",
    "{sub} can be seen as a specialization of {obj}",
    "{obj} generalizes the concept of {sub}",
    "{obj} includes all instances of {sub}",
    "If something is a {sub}, then it is a {obj}",
    "{sub} belongs to the broader class of {obj}",
    "{sub} is more specific than {obj}",
    "{sub} is a manifestation of {obj}",
    "{obj} is a superclass of {sub}",
    "{sub} derives from the category {obj}",
    "{sub} is subsumed by {obj}",
    "{sub} should be classified under {obj}",
    "{sub} expresses all attributes of {obj}",
    "In the context of {obj}, {sub} is a specific case",
    "{sub} is an instantiated form of {obj}",
    "{sub} is a narrower subtype of {obj}",
    "What we call {sub} is just a kind of {obj}",
    "Part of what makes up {obj} is represented by {sub}"
]

_inheritance_templates_q = [
    "Is {sub} a type of {obj}",
    "Is every {sub} an instance of {obj}",
    "Does {sub} fall under the category of {obj}",
    "Can {sub} be seen as a specialization of {obj}",
    "Does {obj} generalize the concept of {sub}",
    "Does {obj} include all instances of {sub}",
    "If something is a {sub}, then is it a {obj}",
    "Does {sub} belong to the broader class of {obj}",
    "Is {sub} more specific than {obj}",
    "Is {sub} a manifestation of {obj}",
    "Is {obj} a superclass of {sub}",
    "Does {sub} derive from the category {obj}",
    "Is {sub} subsumed by {obj}",
    "Should {sub} be classified under {obj}",
    "Does {sub} express all attributes of {obj}",
    "In the context of {obj}, is {sub} a specific case",
    "Is {sub} an instantiated form of {obj}",
    "Is {sub} a narrower subtype of {obj}",
    "Is what we call {sub} just a kind of {obj}",
    "Is part of what makes up {obj} represented by {sub}"
]

_similarity_templates = [
    "{sub} and {obj} are conceptually identical",
    "{sub} is the same as {obj}",
    "{sub} and {obj} refer to the same thing",
    "{sub} equals {obj}",
    "{sub} and {obj} are interchangeable terms",
    "{sub} and {obj} describe the same category",
    "Whether you say {sub} or {obj}, it means the same",
    "{sub} is also known as {obj}",
    "{obj} is an alternative name for {sub}",
    "{sub} and {obj} are equivalent concepts",
    "{sub} and {obj} have no distinction in meaning",
    "People consider {sub} and {obj} to be the same",
    "{sub} and {obj} can substitute for each other",
    "{sub} and {obj} are synonyms",
    "To us, {sub} and {obj} have the same definition",
    "Both {sub} and {obj} signify the same thing",
    "{sub} is recognized as equivalent to {obj}",
    "{sub} and {obj} mutually define one another",
    "{sub} is a valid replacement for {obj}",
    "{sub} and {obj} share a bidirectional ontological relation"
]

_similarity_templates_q = [
    "Are {sub} and {obj} conceptually identical",
    "Is {sub} the same as {obj}",
    "Do {sub} and {obj} refer to the same thing",
    "Does {sub} equal {obj}",
    "Are {sub} and {obj} interchangeable terms",
    "Do {sub} and {obj} describe the same category",
    "Whether you say {sub} or {obj}, does it mean the same",
    "Is {sub} also known as {obj}",
    "Is {obj} an alternative name for {sub}",
    "Are {sub} and {obj} equivalent concepts",
    "Do {sub} and {obj} have no distinction in meaning",
    "Do people consider {sub} and {obj} to be the same",
    "Can {sub} and {obj} substitute for each other",
    "Are {sub} and {obj} synonyms",
    "To us, do {sub} and {obj} have the same definition",
    "Do Both {sub} and {obj} signify the same thing",
    "Is {sub} recognized as equivalent to {obj}",
    "Do {sub} and {obj} mutually define one another",
    "Is {sub} a valid replacement for {obj}",
    "Do {sub} and {obj} share a bidirectional ontological relation"
]


def get_truth_label(freq, truth_categories):
    for lower, upper, labels in truth_categories:
        if lower <= freq < upper:
            return random.choice(labels)


def render_input(task: Task, inheritance_templates, similarity_templates, truth_categories):
    evidence_str = ", ".join(map(str, task.eb))
    truth_label = get_truth_label(task.f, truth_categories)

    if task.copula == "-->":
        template = random.choice(inheritance_templates)
        base_sentence = template.format(sub=task.sub, obj=task.obj)
        return f"{base_sentence}. This statement {truth_label}, based on evidence from judgments {{{evidence_str}}}."
    elif task.copula == "<->":
        template = random.choice(similarity_templates)
        base_sentence = template.format(sub=task.sub, obj=task.obj)
        return f"{base_sentence}. This statement {truth_label}, based on evidence from judgments {{{evidence_str}}}."


def render_question(task: Task, inheritance_templates_q, similarity_templates_q):
    if task.copula == "-->":
        template = random.choice(inheritance_templates_q)
        base_sentence = template.format(sub=task.sub, obj=task.obj)
        return base_sentence
    elif task.copula == "<->":
        template = random.choice(similarity_templates_q)
        base_sentence = template.format(sub=task.sub, obj=task.obj)
        return base_sentence


def parse_output(task_1: Task, task_2: Task, results: [Task, ...]):
    ret = {"premise_1": task_1.to_json(),
           "premise_2": task_2.to_json(),
           "results": [each.to_json() for each in results]}
    return ret


class RuleGrid:

    def __init__(self):
        self.x_axis = ["MP", "PM", "M<>P"]
        self.y_axis = ["SM", "MS", "S<>M"]

    def select_premise_0(self):
        if random.random() < 0.5:
            return random.sample(self.x_axis, 1), random.sample(self.y_axis, 2)
        else:
            return random.sample(self.x_axis, 2), random.sample(self.y_axis, 1)

    @staticmethod
    def select_premise_1(premise_As, premise_Bs):
        return random.choice(premise_As), random.choice(premise_Bs)


def gen_random_reasoning(n=1,
                         inheritance_templates=None, inheritance_templates_q=None,
                         similarity_templates=None, similarity_templates_q=None,
                         truth_categories=None,
                         model_index=0, num_models=1, uniform_sampling=False):
    cases = ["MP, SM", "PM, SM", "M<>P, SM", "MP, MS", "PM, MS", "M<>P, MS", "MP, S<>M", "PM, S<>M", "M<>P, S<>M"]

    if truth_categories is None:
        truth_categories = _truth_categories
    if inheritance_templates is None:
        inheritance_templates = _inheritance_templates
        inheritance_templates_q = _inheritance_templates_q
    if similarity_templates is None:
        similarity_templates = _similarity_templates
        similarity_templates_q = _similarity_templates_q

    if not uniform_sampling:
        accessible_cases = []
        CASES = copy.deepcopy(cases)
        random.shuffle(CASES)
        for _ in range(num_models - 1):
            tmp = []
            for _ in range(len(CASES) // num_models):
                tmp.append(CASES.pop())
            accessible_cases.append(tmp)
        accessible_cases += [CASES]
        accessible_case = accessible_cases[model_index]
    else:
        accessible_case = cases

    rg = RuleGrid()
    ret = []

    for _ in range(n):

        while True:
            S = f"ID_{random.randint(0, 100000)}"
            M = f"ID_{random.randint(0, 100000)}"
            P = f"ID_{random.randint(0, 100000)}"
            if len({S, M, P}) == 3:
                break

        while True:
            premise_As, premise_Bs = rg.select_premise_0()
            premise_A, premise_B = rg.select_premise_1(premise_As, premise_Bs)
            case = f"{premise_A}, {premise_B}"
            if case in accessible_case:
                break

        if case == cases[0]:
            # MP, SM
            J1 = Task(M, P, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(S, M, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[1]:
            # PM, SM
            J1 = Task(P, M, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(S, M, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[2]:
            # M<>P, SM
            J1 = Task(M, P, "<->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(S, M, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[3]:
            # MP, MS
            J1 = Task(M, P, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(M, S, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[4]:
            # PM, MS
            J1 = Task(P, M, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(M, S, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[5]:
            # M<>P, MS
            J1 = Task(M, P, "<->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(M, S, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[6]:
            # MP, S<>M
            J1 = Task(M, P, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(S, M, "<->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[7]:
            # PM, S<>M
            J1 = Task(P, M, "-->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(S, M, "<->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)
        elif case == cases[8]:
            # M<>P, S<>M
            J1 = Task(M, P, "<->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            J2 = Task(S, M, "<->", Truth(random.random(), 0.9),
                      {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
            Rs = random.sample(reasoning(J1, J2), 1)

        ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                     render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                     render_question(Rs[0], inheritance_templates_q, similarity_templates_q),
                     parse_output(J1, J2, Rs)],
                    [J1, J2, Rs]])

    return ret


class Generator:

    def __init__(self, random_seed):
        random.seed(random_seed)

    @staticmethod
    def gen_random_reasoning(n=1,
                             inheritance_templates=None, inheritance_templates_q=None,
                             similarity_templates=None, similarity_templates_q=None,
                             truth_categories=None,
                             model_index=0, num_models=1, uniform_sampling=False):
        return gen_random_reasoning(n,
                                    inheritance_templates, inheritance_templates_q,
                                    similarity_templates, similarity_templates_q,
                                    truth_categories,
                                    model_index, num_models, uniform_sampling)


if __name__ == "__main__":
    G = Generator(39)
    samples = G.gen_random_reasoning(num_models=3, model_index=0)
    # samples = G.gen_random_reasoning(uniform_sampling=True)
    print(samples)
