import heapq
# use dataclasses to improve readability
from dataclasses import dataclass, field
from itertools import count
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# TODO: Unterschied SC und CC einfügen
# TODO: Unterschiedliche Kapazitäten einfügen
# TODO: Unterschiedliches handling in SC und CC einfügen


@dataclass(slots=True)
class Customer:
    """class to keep track of customers"""

    # initialize Variables for Object
    t_arr: float
    """ arrival time of the customer """
    cust_id: int = field(default_factory=count(start=1).__next__, init=False)
    """ generate an unique id for every customer """
    t_dep: float = 0
    """ initialize departure time """
    t_proc: float = 0
    """ initialize processing time """


@dataclass(slots=True, frozen=True)
class Event:
    """class for keeping track of events"""

    # initialize variables for object
    # init is set to false, so the counter doesn't reset every time
    ev_id: int = field(default_factory=count(start=1).__next__, init=False)
    """ generate a new id when a class object is created, used for sorting purposes in heapq """
    time: float
    """ time at which event occurs """
    kind: str
    """ kind of event "arr": arrival and "dep": departure """
    customer: object
    """ customer that is handled """
    c_id: int
    """ id of checkout where event occurs """


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
    c_status: int = 0
    """ status of the cashier"""
    ql: int = 0
    """queue length of the checkout"""
    c_quant: int = 1
    """ number of cashiers """
    sc_quant: int = 6
    # TODO: Durch diese Implementierung parameter ql unnötig?
    # Optimierung später!
    queue: List[Tuple[float, int, Customer]] = field(default_factory=list)
    """ list of costumers in queue"""

    # heapify the customer list, so
    def __post_init__(self):
        heapq.heapify(self.queue)


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
        heapq.heappush(self.event_list, (t_arrival, new_arr.ev_id, new_arr))

    # use ql for checking
    # TODO: länge des checkout queue an sich nehmen und nicht einfach nur nummer speichern
    def update_ql(self):
        new_ql = []
        for i in range(self.sum_c):
            new_ql.append(len(self.checkouts[f"Checkout{i + 1}"].queue))
        self.queue_log.append((self.t, [new_ql[x] for x in range(self.sum_c)]))
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
                current_event.customer.cust_id,
                current_event.c_id,
            ]
        )
        checkout = self.checkouts[f"Checkout{current_event.c_id}"]
        # push customer into queue
        heapq.heappush(
            self.checkouts[f"Checkout{current_event.c_id}"].queue,
            (
                current_event.customer.t_arr,
                current_event.customer.cust_id,
                current_event.customer,
            ),
        )
        self.update_ql()
        # if checkout is idle schedule departure event
        if checkout.c_status == 0:
            # generate processing time
            proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            # calculate new time
            t_1 = self.t + proc_rate
            # create new event
            new_dep = Event(t_1, "dep", current_event.customer, current_event.c_id)
            # add new departure event
            heapq.heappush(self.event_list, (t_1, new_dep.ev_id, new_dep))
            # add departure and processing time for customer
            current_event.customer.t_dep = t_1
            current_event.customer.t_proc = proc_rate
            # TODO: 1. hier noch möglichkeit für sc einfügen, dass mehr Kapazität vorhanden ist
            # TODO: 2. hier wird  Kapazität bis jetzt auf 1 gesetzt
            # set c_status to 1 for busy
            checkout.c_status = 1
        # generate new arrival event
        self.get_arrival()

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
                current_event.customer.cust_id,
                current_event.c_id,
            ]
        )
        # get checkout object from dict
        checkout = self.checkouts[f"Checkout{current_event.c_id}"]
        # set checkout status to idle
        checkout.c_status = 0
        # pop leaving customer from respective queue
        heapq.heappop(checkout.queue)

        if len(checkout.queue) > 0:
            # generate random processing time
            proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            # calculate departure time
            t_dep = self.t + proc_rate
            # get customer from checkout queue
            dep_customer = checkout.queue[0][2]
            # set departure and processing time for log
            dep_customer.t_dep = t_dep
            # create new departure event
            new_dep = Event(t_dep, "dep", dep_customer, checkout.c_id)
            # create new departure event
            heapq.heappush(self.event_list, (t_dep, new_dep.ev_id, new_dep))
            # set cashier status to busy
            checkout.c_status = 1

        # log customer
        self.customer_log.append(
            [
                current_event.customer.cust_id,
                current_event.customer.t_arr,
                current_event.customer.t_dep,
                current_event.customer.t_proc,
            ]
        )

    def next_action(self):
        if self.event_list[0][2].kind == "arr":
            self.arrival()
        elif self.event_list[0][2].kind == "dep":
            self.departure()

    # TODO: Methode einfügen, die Simulation durchführt und als Eingabe Zahl bekommt,
    #  wie oft Simulation durchgeführt werden soll -> erst später als Feature!
    # Als statische Methode implementieren, die Dict mit einzelnen Parametern
    # für die Simulationen bekommen soll? Also nicht nur festlegen wie oft, sondern auch
    # unterschiedliche Parameter möglich?
    # Dictionary: Keys -> Name der Simulation, Values -> Liste der Keyword argumente
    # mit parametern
    def simulate(self):
        """
        :return: Event-, Customer- und queue Log als Liste
        """

        # 1. Get initial Arrival
        self.get_arrival()
        # 2. process events until time limit is reached
        # TODO: Funktionen ober so umbauen, dass queue- und log updates in dieser Funktion stattfinden?
        with tqdm(total=self.t_max, unit_scale=True) as pbar:
            while self.t < self.t_max:
                t_old = float(self.t)
                self.next_action()
                t_delta = float(self.t) - t_old
                pbar.update(t_delta)

        return self.event_log, self.customer_log, self.queue_log


def main():
    my_sim = Simulation(arrival_rate=5)
    ev_log, c_log, q_log = my_sim.simulate()

    ev_df = pd.DataFrame(ev_log)
    ev_df.columns = ["event_id", "time", "kind", "customer_id", "checkout_id"]
    c_df = pd.DataFrame(c_log)
    c_df.columns = ["customer_id", "arrival_time", "departure_time", "processing_rate"]
    # TODO: Header schick machen und Werte vernünftig einsortieren
    q_df = pd.DataFrame(q_log)
    q_df.columns = ["time", [key for key in my_sim.checkouts.keys()]]

    ev_df.to_csv("event_log.csv", index=False)
    c_df.to_csv("customer_log.csv", index=False)
    q_df.to_csv("queue_log.csv", index=False)


# signalize to reader of code that this is a script and not just a library
if __name__ == "__main__":
    main()
