import heapq
# use dataclasses to improve readability
from dataclasses import dataclass, field
from itertools import count
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# TODO: Proc Rate als Variable einfügen
# TODO: Arrival Rate als Variable einfügen!


@dataclass(slots=True)
class Customer:
    """class to keep track of customers"""

    # initialize Variables for Object
    t_arr: float
    """ arrival time of the customer """
    cust_id: int = field(default_factory=count().__next__, init=False)
    """ generate an unique id for every customer """
    t_dep: float = 0
    """ initialize departure time """


@dataclass(slots=True, frozen=True)
class Event:
    """class for keeping track of events"""

    # initialize variables for object
    # init is set to false, so the counter doesn't reset every time
    ev_id: int = field(default_factory=count().__next__, init=False)
    """ generate a new id when a class object is created, used for sorting purposes in heapq """
    time: float
    """ time at which event occurs """
    kind: str
    """ kind of event "arr": arrival and "dep": departure """
    customer: object
    """ customer that is handled """
    c_id: int
    """ id of checkout where event occurs """

    # define methods
    # unnötige methode?
    # def get_event_data(self) -> Tuple[float, str, int]:
    #     """
    #     :return: return event data for the event list
    #     """
    #     return self.time, self.kind, self.c_id


# welchen Datentyp hat eigentlich Zeit? Int? → erstmal float nutzen
# Everything is stored in one event list!


@dataclass(slots=True)
class Checkout:
    """Class to keep track of Checkouts"""

    # initialize variables
    c_id: int
    """ id of the checkout """
    c_type: str
    """ type of the checkout"""
    c_status = 0
    """ status of the cashier"""
    ql: int = 0
    """queue length of the checkout"""
    c_quant: int = 1
    """ number of cashiers """
    sc_quant: int = 6
    # TODO: Durch diese Implementierung parameter ql unnötig?
    # Optimierung später!
    queue: List[Tuple[float, Customer]] = field(default_factory=list)
    """ list of costumers in queue"""

    # heapify the customer list, so
    def __post_init__(self):
        heapq.heapify(self.queue)


# TODO: Customer Log einfügen -> bei jedem departure einfach customer übergeben?
# TODO: Event Log einfügen -> bei jedem event pop einfach event in liste speichern?
class Simulation:
    def __init__(
            self,
            s_seed: int = 42,
            t_max: float = 1000,
            proc_rate_cc: int = 1,
            proc_rate_sc: int = 1,
            arrival_rate: int = 1,
            num_cc: int = 6,
            num_sc: int = 1,
    ):
        # TODO: Checken ob so korrekt implementiert
        # initialize Values
        # initialize processing rates
        self.processing_rate = {"cc": proc_rate_cc, "sc": proc_rate_sc}
        self.arrival_rate = arrival_rate
        self.num_cc = num_cc
        self.num_sc = num_sc
        self.sum_c = num_cc + num_sc
        self.t_max = t_max
        self.t = 0
        self.ql_list = []
        self.event_list = []
        heapq.heapify(self.event_list)
        self.rng = np.random.default_rng(seed=s_seed)
        self.checkouts = {}
        self.event_log = []
        self.customer_log = []
        self.queue_log = []

        for i in range(self.num_cc):
            num = i + 1
            key = f"Checkout{num}"
            self.checkouts[key] = Checkout(num, "cc")
        for j in range(self.num_cc, self.sum_c):
            num = j + 1
            key = f"Checkout{num}"
            self.checkouts[key] = Checkout(num, "sc")
        # Initialize the ql_list
        self.update_ql()

    def get_arrival(self):
        # calculate the min value in list
        min_value = min(self.ql_list)
        # search for max in list and return indices
        min_index = [i for i, m in enumerate(self.ql_list) if m == min_value]
        # choose randomly from list and increment by 1, since Checkouts start at 1
        c_id = self.rng.choice(min_index) + 1
        # TODO: nochmal checken ob hier richtig!
        t_arrival = self.t + self.rng.exponential(self.arrival_rate)
        # create new customer
        new_customer = Customer(t_arrival)
        # create new event
        new_arr = Event(t_arrival, "arr", new_customer, c_id)
        # push event into heapq, use ev_id for sorting if events occur at same time
        # and c_id to schedule new events
        heapq.heappush(
            self.event_list, (t_arrival, new_arr.ev_id, new_arr.c_id, new_arr)
        )
        # Customer stellt sich doch erst zum Zeitpunkt selber an, oder? -> in Implementierung egal, da log != ql
        # und über ql Kasse ausgewählt wird.
        # ABER: kann man auch anders implementieren. Dann muss Codeblock doch nur in Arrival geschoben werden,
        # oder nicht?
        # Dann stellt sich Customer bei Arrival an
        # TODO: Checken ob richtig, siehe Kommentar Zeile drüber
        # put customer into checkout queue
        heapq.heappush(
            self.checkouts[f"Checkout{new_arr.c_id}"].queue,
            (new_customer.t_arr, new_customer),
        )
        self.update_ql()

    # use ql for checking
    def update_ql(self):
        new_ql = []
        for i in range(self.sum_c):
            new_ql.append(self.checkouts[f"Checkout{i + 1}"].ql)
        self.queue_log.append((self.t, new_ql))
        self.ql_list = new_ql

    # TODO: Unterschied SC und CC in Arrival implementieren
    def arrival(self):
        # grab first event from heapq
        self.t, ev_id, current_event = heapq.heappop(self.event_list)
        # log event
        self.event_log.append(
            [
                current_event.ev_id,
                current_event.time,
                current_event.kind,
                current_event.customer,
                current_event.c_id,
            ]
        )
        checkout = self.checkouts[f"Checkout{current_event.c_id}"]
        # if checkout is idle schedule departure event
        if checkout.c_status == 0:
            # generate processing time
            proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            # calculate new time
            t_1 = self.t + proc_rate
            # create new event
            new_dep = Event(t_1, "dep", current_event.customer, current_event.c_id)
            # add new departure event
            heapq.heappush(self.event_list, (t_1, new_dep.ev_id, new_dep.c_id, new_dep))
            # add departure time for customer
            current_event.customer.dep_time = t_1
            # TODO: 1. hier noch möglichkeit für sc einfügen, dass mehr Kapazität vorhanden ist
            # TODO: 2. hier wird  Kapazität bis jetzt auf 1 gesetzt
            # set c_status to 1 for busy
            checkout.c_status = 1
        else:
            checkout.ql += 1
        # generate new arrival event
        self.get_arrival()
        # Legacy Code ersetzt durch funktion drüber
        # inter_time = self.rng.exponential(self.arrival_rate)
        # arr_time = inter_time + time
        # heapq.heappush(
        #     self.event_list,
        #     Event(
        #         arr_time,
        #         "arr",
        #         Customer(arr_time),
        #     ),
        # )
        raise NotImplementedError

    # TODO: Unterschied SC und CC in departure implementieren
    def departure(self):
        # pop from heap
        self.t, ev_id, current_event = heapq.heappop(self.event_list)
        # log event
        self.event_log.append(
            [
                current_event.ev_id,
                current_event.time,
                current_event.kind,
                current_event.customer,
                current_event.c_id,
            ]
        )
        # get checkout object from dict
        checkout = self.checkouts[f"Checkouts{current_event.c_id}"]
        # set checkout status to idle
        checkout.c_status = 0
        # check if events in queue
        # TODO: nächstes event in der queue an der Kasse abgreifen und departure planen
        if checkout.ql > 0:
            # generate random processing time
            proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            # calculate departure time
            t_dep = self.t + proc_rate
            # pop Customer from checkout queue
            _, dep_customer = heapq.heappop(checkout.queue)
            # set departure time for log
            dep_customer.t_dep = t_dep
            # create new departure event
            new_dep = Event(t_dep, "dep", dep_customer, checkout.c_id)
            # create new departure event
            heapq.heappush(
                self.event_list, (t_dep, new_dep.ev_id, new_dep.c_id, new_dep)
            )
            # set cashier status to busy
            checkout.c_status = 1
            # decrease queue length
            checkout.ql -= 1
            # log customer
            self.customer_log.append(
                [
                    dep_customer.cust_id,
                    dep_customer.t_arr,
                    dep_customer.t_dep,
                    proc_rate,
                ]
            )
        raise NotImplementedError

    def next_action(self):
        if self.event_list[0].kind == "arr":
            self.arrival()
        elif self.event_list[0].kind == "dep":
            self.departure()

    # TODO: Methode einfügen, die Simulation durchführt und als Eingabe Zahl bekommt,
    #  wie oft Simulation durchgeführt werden soll -> erst später als Feature!
    # Als statische Methode implementieren, die Dict mit einzelnen Parametern
    # für die Simulationen bekommen soll? Also nicht nur festlegen wie oft, sondern auch
    # unterschiedliche Parameter möglich?
    # Dictionary: Keys -> Name der Simulation, Values -> Liste der Keywordargumente mit parametern
    def simulate(self):
        """
        :return: Event-, Customer- und queue Log als Liste
        """

        # 1. Get initial Arrival
        self.get_arrival()
        # 2. process events until time limit is reached
        # TODO: Funktionen ober so umbauen, dass queue- und log updates in dieser Funktion stattfinden
        pbar = tqdm(total=self.t_max + 1)
        while self.t < self.t_max:
            self.next_action()
            pbar.update(1)

        return self.event_log, self.customer_log, self.queue_log


def main():
    my_sim = Simulation()
    ev_log, c_log, q_log = my_sim.simulate()

    ev_df = pd.DataFrame(ev_log)
    c_df = pd.DataFrame(c_log)
    q_df = pd.DataFrame(q_log)

    ev_df.to_csv("event_log.csv")
    c_df.to_csv("customer_log.csv")
    q_df.to_csv("queue_log")

    # Legacy Code
    # t = 0
    # pbar = tqdm(total=100000 + 1)
    # while t <= 100000:
    #     t += 1
    #     pbar.update(1)
    # pbar.close()
    # return print("Done")


# signalize to reader of code that this is a script and not just a library
if __name__ == "__main__":
    main()

###############
# Legacy Code #
###############
# def simulate(
#     num_cc: int,
#     num_sc: int,
#     t_max: float,
#     # define number of simulations in function or create a for loop outside?
#     run_num: int = 1,
# ) -> None:
#     # store nr. of checkouts
#     sum_c = num_cc + num_sc
#     # initialize rng generator
#     rng = np.random.default_rng(seed=42)
#     # Initialize Event list
#     event_list: list[Union[int, Event]] = []
#     heapq.heapify(event_list)
#     # Initialize time
#     t = 0.0
#     # initialize dictionary that holds the Checkout object
#     checkouts = {}
#
#
#     get_arrival(event_list, ql_list, rng, t)
#     # method to advance time
#     while t < t_max:
#         t = t + 1
#         pass
#
#     # print loop for testing purposes
#     for i in range(num_cc + num_sc):
#         print("------------------------------")
#         print(f"Checkout{i + 1} = {checkouts[f'Checkout{i + 1}']}")
#
#     print(event_list)
#
#
# simulation(3, 1, 100)
