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
        t = Task(task_2.obj, task_1.sub, "-->", TFS.exe_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "exe_p")
        ret.append(t)
    elif task_1.obj == task_2.obj and task_1.copula == task_2.copula == "-->":
        # PM, SM -> SP(abd), PS('abd), S<>P('com)
        t = Task(task_2.sub, task_1.sub, "-->", TFS.abd(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "abd")
        ret.append(t)
        t = Task(task_1.sub, task_2.sub, "-->", TFS.abd_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "adb_p")
        ret.append(t)
        t = Task(task_2.sub, task_1.sub, "<->", TFS.com_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com_p")
        ret.append(t)
        t = Task(task_1.sub, task_2.sub, "<->", TFS.com_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com_p")
        ret.append(t)
    elif task_1.sub == task_2.obj and task_1.copula == "<->" and task_2.copula == "-->":
        # M<>P, SM -> SP('ana)
        t = Task(task_2.sub, task_1.obj, "-->", TFS.ana_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ana_p")
        ret.append(t)
    elif task_1.sub == task_2.sub and task_1.copula == task_2.copula == "-->":
        # MP, MS -> SP(ind), PS('ind), S<>P(com)
        t = Task(task_2.obj, task_1.obj, "-->", TFS.ind(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ind")
        ret.append(t)
        t = Task(task_1.obj, task_2.obj, "-->", TFS.ind_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ind_p")
        ret.append(t)
        t = Task(task_2.obj, task_1.obj, "<->", TFS.com(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com")
        ret.append(t)
        t = Task(task_1.obj, task_2.obj, "<->", TFS.com(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "com")
        ret.append(t)
    elif task_1.obj == task_2.sub and task_1.copula == task_2.copula == "-->":
        # PM, MS -> SP(exe), PS('ded)
        t = Task(task_2.obj, task_1.sub, "-->", TFS.exe(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "exe")
        ret.append(t)
        t = Task(task_1.sub, task_2.obj, "-->", TFS.ded_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ded_p")
        ret.append(t)
    elif task_1.sub == task_2.sub and task_1.copula == "<->" and task_2.copula == "-->":
        # M<>P, MS -> PS('ana)
        t = Task(task_1.obj, task_2.obj, "-->", TFS.ana_p(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ana_p")
        ret.append(t)
    elif task_1.sub == task_2.obj and task_1.copula == "-->" and task_2.copula == "<->":
        # MP, S<>M -> SP(ana)
        t = Task(task_2.sub, task_1.obj, "-->", TFS.ana(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ana")
        ret.append(t)
    elif task_1.obj == task_2.obj and task_1.copula == "-->" and task_2.copula == "<->":
        # PM, S<>M -> PS(ana)
        t = Task(task_1.sub, task_2.sub, "-->", TFS.ana(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "ana")
        ret.append(t)
    elif task_1.sub == task_2.obj and task_1.copula == task_2.copula == "<->":
        # M<>P, S<>M -> S<>P(res)
        t = Task(task_2.sub, task_1.obj, "<->", TFS.res(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "res")
        ret.append(t)
        t = Task(task_1.obj, task_2.sub, "<->", TFS.res(task_1.truth, task_2.truth), task_1.eb.union(task_2.eb), "res")
        ret.append(t)

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


def parse_output(task_1: Task, task_2: Task, results: [Task, ...]):
    ret = {"premise_1": task_1.to_json(),
           "premise_2": task_2.to_json(),
           "results": [each.to_json() for each in results]}
    return ret


def gen_random_reasoning(n=1, inheritance_templates=None, similarity_templates=None, truth_categories=None):

    if truth_categories is None:
        truth_categories = _truth_categories
    if similarity_templates is None:
        similarity_templates = _similarity_templates
    if inheritance_templates is None:
        inheritance_templates = _inheritance_templates

    ret = []
    cases = ["MP, SM", "PM, SM", "M<>P, SM", "MP, MS", "PM, MS", "M<>P, MS", "MP, S<>M", "PM, S<>M", "M<>P, S<>M"]
    for _ in range(n):
        case = random.choice(range(len(cases)))
        S = f"ID_{random.randint(0, 100000)}"
        M = f"ID_{random.randint(0, 100000)}"
        P = f"ID_{random.randint(0, 100000)}"
        if len({S, M, P}) == 3:
            if case == 0:
                # MP, SM
                J1 = Task(M, P, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(S, M, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 1:
                # PM, SM
                J1 = Task(P, M, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(S, M, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 2:
                # M<>P, SM
                J1 = Task(M, P, "<->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(S, M, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories), parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 3:
                # MP, MS
                J1 = Task(M, P, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(M, S, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 4:
                # PM, MS
                J1 = Task(P, M, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(M, S, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 5:
                # M<>P, MS
                J1 = Task(M, P, "<->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(M, S, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 6:
                # MP, S<>M
                J1 = Task(M, P, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(S, M, "<->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 7:
                # PM, S<>M
                J1 = Task(P, M, "-->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(S, M, "<->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
            elif case == 8:
                # M<>P, S<>M
                J1 = Task(M, P, "<->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                J2 = Task(S, M, "<->", Truth(random.random(), 0.9),
                          {random.randint(0, 10000) for _ in range(random.randint(1, 2))})
                Rs = reasoning(J1, J2)
                ret.append([[render_input(J1, inheritance_templates, similarity_templates, truth_categories),
                             render_input(J2, inheritance_templates, similarity_templates, truth_categories),
                             parse_output(J1, J2, Rs)], [J1, J2, Rs]])
    return ret


if __name__ == "__main__":
    t1 = Task("A", "B", "-->", Truth(), {0})
    t2 = Task("B", "C", "-->", Truth(), {1})
    samples = gen_random_reasoning()
    print(samples)
