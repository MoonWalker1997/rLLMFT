import random

import numpy as np

from config import cs, _inheritance_templates, _inheritance_templates_q, _similarity_templates, _similarity_templates_q, \
    _truth_categories


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
        return self.c / (1 - self.c)

    @property
    def wp(self):
        return self.w * self.f

    @property
    def wn(self):
        return self.w * (1 - self.f)

    def __str__(self):
        return f"%{round(self.f, 3)};{round(self.c, 3)}%"


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
        return f"{base_sentence}. This statement {truth_label}, based on evidence from evidence {{{evidence_str}}}."
    elif task.copula == "<->":
        template = random.choice(similarity_templates)
        base_sentence = template.format(sub=task.sub, obj=task.obj)
        return f"{base_sentence}. This statement {truth_label}, based on evidence from evidence {{{evidence_str}}}."


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


def instantiate_and_reasoning(S, M, P, case, J1=None, J2=None):
    Rs = []

    while True:
        eb1 = {random.randint(0, 10000) for _ in range(random.randint(1, 2))} if not J1 else J1.eb
        eb2 = {random.randint(0, 10000) for _ in range(random.randint(1, 2))} if not J2 else J2.eb
        if eb1 != eb2:
            break

    if case == "MP, SM":
        J1 = J1 or Task(M, P, "-->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(S, M, "-->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "PM, SM":
        J1 = J1 or Task(P, M, "-->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(S, M, "-->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "M<>P, SM":
        J1 = J1 or Task(M, P, "<->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(S, M, "-->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "MP, MS":
        J1 = J1 or Task(M, P, "-->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(M, S, "-->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "PM, MS":
        J1 = J1 or Task(P, M, "-->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(M, S, "-->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "M<>P, MS":
        J1 = J1 or Task(M, P, "<->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(M, S, "-->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "MP, S<>M":
        J1 = J1 or Task(M, P, "-->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(S, M, "<->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "PM, S<>M":
        J1 = J1 or Task(P, M, "-->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(S, M, "<->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)
    elif case == "M<>P, S<>M":
        J1 = J1 or Task(M, P, "<->", Truth(random.random(), 0.9), eb1)
        J2 = J2 or Task(S, M, "<->", Truth(random.random(), 0.9), eb2)
        Rs = reasoning(J1, J2)

    return J1, J2, Rs


def gen_random_reasoning(cases,
                         n,
                         inheritance_templates, inheritance_templates_q,
                         similarity_templates, similarity_templates_q,
                         truth_categories,
                         model_index=0, num_models=1, for_testing=False):
    # different models (marked by indices) are trained using different rules to simulate the bias
    # though tested with all rules, when for_testing is true

    if not for_testing:
        # split all cases with no overlapping
        shuffled = list(cases)
        random.shuffle(shuffled)
        accessible_case = np.array_split(shuffled, num_models)[model_index].tolist()
    else:
        accessible_case = cases

    ret = []
    for _ in range(n):

        # find a case for step-1 reasoning
        case_1 = random.choice(accessible_case)
        S_1, M_1, P_1 = [f"ID_{each}" for each in random.sample(range(0, 100000), 3)]
        J1_1, J2_1, Rs_1 = instantiate_and_reasoning(S_1, M_1, P_1, case_1)
        Rs_1 = random.choice(Rs_1)

        # fine J1 distracting premises
        dt_case_1 = random.choice(accessible_case)  # dt for distracting
        while True:
            # avoiding as the same as the non-distracting one (though very unlikely)
            dt_S_1, dt_M_1, dt_P_1 = [f"ID_{each}" for each in random.sample(range(0, 100000), 3)]
            if dt_S_1 != S_1 or dt_M_1 != M_1 or dt_P_1 != P_1:
                break
        dt_J1_1, dt_J2_1, _ = instantiate_and_reasoning(dt_S_1, dt_M_1, dt_P_1, dt_case_1)

        # >--
        # go for step-2 reasoning

        # get SMP/case for step-2
        while True:
            S_2, M_2, P_2 = [f"ID_{each}" for each in random.sample(range(0, 100000), 3)]
            if len({S_1, M_1, P_1, S_2, M_2, P_2}) == 6:
                break
        d = {"S": S_2, "M": M_2, "P": P_2}
        random.shuffle(accessible_case)
        mk = -1
        for each_case in accessible_case:
            if mk != -1:
                break
            for i, each in enumerate(each_case.split(", ")):
                if ("<>" in each and Rs_1.copula != "<->") or ("<>" not in each and Rs_1.copula != "-->"):
                    continue
                each = each.replace("<>", "")
                d[each[0]], d[each[1]] = Rs_1.sub, Rs_1.obj
                mk = i
                case_2 = each_case
                break

        if mk == -1:
            continue
        elif mk == 0:
            J1_2 = Rs_1
            _, J2_2, Rs_2 = instantiate_and_reasoning(d["S"], d["M"], d["P"], case_2, J1=Rs_1)
        elif mk == 1:
            J2_2 = Rs_1
            J1_2, _, Rs_2 = instantiate_and_reasoning(d["S"], d["M"], d["P"], case_2, J2=Rs_1)

        Rs_2 = random.choice(Rs_2)

        premises_texts = [render_input(J1_1, inheritance_templates, similarity_templates, truth_categories),
                          render_input(J2_1, inheritance_templates, similarity_templates, truth_categories),
                          render_input(dt_J1_1, inheritance_templates, similarity_templates, truth_categories),
                          render_input(dt_J2_1, inheritance_templates, similarity_templates, truth_categories)]
        if mk == 0:
            premises_texts.append(render_input(J2_2, inheritance_templates, similarity_templates, truth_categories))
        if mk == 1:
            premises_texts.append(render_input(J1_2, inheritance_templates, similarity_templates, truth_categories))

        question = render_question(Rs_2, inheritance_templates_q, similarity_templates_q)
        results_jsons = [parse_output(J1_1, J2_1, [Rs_1]),
                         parse_output(J1_2, J2_2, [Rs_2])]
        ret.append([premises_texts, question, results_jsons])

    return ret


class Generator:

    def __init__(self, random_seed):
        random.seed(random_seed)

    @staticmethod
    def gen_random_reasoning(cases,
                             n=1,
                             inheritance_templates=None, inheritance_templates_q=None,
                             similarity_templates=None, similarity_templates_q=None,
                             truth_categories=None,
                             model_index=0, num_models=1, uniform_sampling=False):
        return gen_random_reasoning(cases,
                                    n,
                                    inheritance_templates, inheritance_templates_q,
                                    similarity_templates, similarity_templates_q,
                                    truth_categories,
                                    model_index, num_models, uniform_sampling)


if __name__ == "__main__":
    G = Generator(39)
    samples = G.gen_random_reasoning(cs,
                                     1,
                                     _inheritance_templates, _inheritance_templates_q,
                                     _similarity_templates, _similarity_templates_q,
                                     _truth_categories,
                                     model_index=0, num_models=1, uniform_sampling=False)
    print(samples)
