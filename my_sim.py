import heapq
import warnings
# use dataclasses to improve readability
from dataclasses import dataclass, field
from itertools import count
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# ignore Warning, that the progress bar might go slightly over 100% due to rounding error
warnings.filterwarnings("ignore", module="tqdm")


# TODO: proc_num durch len(checkout.processing) ersetzen
# TODO: Visualisierungen erstellen
# TODO: Experiment Design festlegen (einfach unterschiedliche Parameter nutzen und dann Plots machen und vergleichen)
# TODO: Tendenz SC/CC zu nutzen pro Kunde generieren
# TODO: Verteilungen für Service Time abstrahieren
# TODO: Normalverteilung für Service Time?
# TODO: unterschiedliche Service Time für SC vs CC (SC halb so schnell?) -> über Verteilungen geregelt, aber
# vielleicht für generelle implementierung noch wichtig
# TODO: unterschiedliche Verteilungen einbauen


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
    """ kind of event ("arr": arrival and "dep": departure) """
    customer: object
    """ customer that is handled """
    c_id: int
    """ id of checkout where event occurs """


@dataclass(slots=True)
class Customer:
    """class to keep track of customers"""

    # initialize Variables for Object
    t_arr: float
    """ arrival time of the customer """
    cust_id: int = field(default_factory=count(start=1).__next__, init=False)
    """ generate an unique id for every customer """
    t_dep: float = None
    """ initialize departure time """
    t_proc: float = None
    """ initialize processing time """
    num_items: int = None
    """ number of items the customer wants to buy"""


@dataclass(slots=True)
class Checkout:
    """Class to keep track of Checkouts"""

    # initialize variables
    c_id: int
    """ id of the checkout """
    c_type: str
    """ type of the checkout """
    # initialisierung von c_status und proc_num in post init, da keine Variable?
    c_status: int = 0
    """ status of the cashier"""
    c_quant: int = 1
    """ number of cashiers """
    sc_quant: int = 6
    """ number of self checkouts """
    proc_num: int = 0
    """ number of customers in processing """
    queue: List[Tuple[float, int, Customer]] = field(default_factory=list)
    """ list of costumers in queue"""
    processing: List[Tuple[float, int, Customer]] = field(default_factory=list)
    """ list of customers in processing """

    # heapify the queue and processing list
    def __post_init__(self):
        heapq.heapify(self.queue)
        heapq.heapify(self.processing)


class Simulation:
    """ Class for simulation of a supermarket"""
    def __init__(
            self,
            s_seed: int = 42,
            t_max: float = 1000,
            # TODO: Hier unterschiedliche processing Verteilungen einfügen?
            proc_rate_cc: int = 1,
            proc_rate_sc: int = 1,
            arrival_rate: int = 1,
            num_cc: int = 6,
            num_sc: int = 1,
            c_quant: int = 1,
            sc_quant: int = 6,
            item_scale: float = 14.528563291255535  # Scale parameter for exponential distribution
    ):
        # initialize parameters
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
        self.c_quant = c_quant
        self.sc_quant = sc_quant
        self.item_scale = item_scale

        for i in range(self.num_cc):
            num = i + 1
            key = f"Checkout{num}"
            self.checkouts[key] = Checkout(num, "cc", c_quant=self.c_quant)
        for j in range(self.num_cc, self.sum_c):
            num = j + 1
            key = f"Checkout{num}"
            self.checkouts[key] = Checkout(num, "sc", sc_quant=self.sc_quant)
        # Initialize the ql_list
        self.update_ql()

    def get_arrival(self):
        """ sample arrival time, choose shortest queue, generate a new_customer
        and add the new arrival event to the event list of the class object """
        # calculate the min value in list
        min_value = min(self.ql_list)
        # search for max in list and return indices
        min_index = [i for i, m in enumerate(self.ql_list) if m == min_value]
        # choose randomly from list and increment by 1, since Checkouts start at 1
        c_id = self.rng.choice(min_index) + 1
        # calculate arrival time
        t_arrival = self.t + self.rng.exponential(self.arrival_rate)
        # generate number of items the customer wants to buy
        # scale is from the data analysis part
        # assumption: no fractional items -> round up
        num_items = np.ceil(self.rng.exponential(scale=self.item_scale))
        # create new customer
        new_customer = Customer(t_arrival, num_items=num_items)
        # create new event
        new_arr = Event(t_arrival, "arr", new_customer, c_id)
        # push event into heapq, use ev_id for sorting if events occur at same time
        # and c_id to schedule new events
        heapq.heappush(self.event_list, (t_arrival, new_arr.ev_id, new_arr))

    # use ql for checking
    def update_ql(self):
        """ update the list of queue lengths and write it to the queue log"""
        new_ql = []
        for i in range(self.sum_c):
            new_ql.append(len(self.checkouts[f"Checkout{i + 1}"].queue))
        self.queue_log.append((self.t, [new_ql[x] for x in range(self.sum_c)]))
        self.ql_list = new_ql

    def arrival(self):
        """
        pops the arrival event from the event list (first entry, since min-heap is used), and writes it to event log.
        A customer is added to the respective checkout queue and the update_ql() method is called.
        If the queue is empty the customer is processed ################ Zu ende schreiben  #############
        """
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
            # pop customer from checkout queue
            _, _, proc_customer = heapq.heappop(checkout.queue)
            # generate processing time per item
            # TODO: Parameter nicht hardcoden, sondern übergeben
            if checkout.c_type == 'cc':
                proc_rate_per_item = self.rng.laplace(loc=3.7777777777777777, scale=2.1742325579116906)
            elif checkout.c_type == 'sc':
                proc_rate_per_item = self.rng.gumbel(loc=11.224246069610276, scale=6.208811868891992)
            else:
                raise ValueError('Checkout Type is neither "cc" nor "sc"')
            # calculate processing time for all items
            proc_rate = proc_rate_per_item * proc_customer.num_items
            # calculate new time
            t_1 = self.t + proc_rate
            # create new event
            # TODO: current_event.customer durch proc_customer ersetzen? nochmal abchecken
            new_dep = Event(t_1, "dep", current_event.customer, current_event.c_id)
            # add new departure event
            heapq.heappush(self.event_list, (t_1, new_dep.ev_id, new_dep))
            # add departure and processing time for customer
            current_event.customer.t_dep = t_1
            current_event.customer.t_proc = proc_rate
            # put customer in processing queue
            heapq.heappush(
                checkout.processing,
                (proc_customer.t_dep, proc_customer.cust_id, proc_customer),
            )
            # Increment Customers in Processing
            checkout.proc_num += 1
            # check if capacity of cashier is reached and set status to busy
            if checkout.c_type == "cc":
                if checkout.proc_num == checkout.c_quant:
                    checkout.c_status = 1
            elif checkout.c_type == "sc":
                if checkout.proc_num == checkout.sc_quant:
                    checkout.c_status = 1

        # generate new arrival event
        self.get_arrival()

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
        # decrease Customers in processing
        checkout.proc_num -= 1
        # set checkout status to idle
        checkout.c_status = 0
        # pop leaving customer from respective queue
        heapq.heappop(checkout.processing)

        if len(checkout.queue) > 0:
            # generate random processing time
            # legacy code: proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            if checkout.c_type == 'cc':
                proc_rate_per_item = self.rng.laplace(loc=3.7777777777777777, scale=2.1742325579116906)
            elif checkout.c_type == 'sc':
                proc_rate_per_item = self.rng.gumbel(loc=11.224246069610276, scale=6.208811868891992)
            else:
                raise ValueError('Checkout Type is neither "cc" nor "sc"')
            # get customer from checkout queue
            dep_customer = checkout.queue[0][2]
            # calculate total proc_rate for all items
            proc_rate = dep_customer.num_items * proc_rate_per_item
            # calculate departure time
            t_dep = self.t + proc_rate
            # set departure and processing time for log
            dep_customer.t_dep = t_dep
            # create new departure event
            new_dep = Event(t_dep, "dep", dep_customer, checkout.c_id)
            # create new departure event
            heapq.heappush(self.event_list, (t_dep, new_dep.ev_id, new_dep))
            # Increment Customers in Processing
            checkout.proc_num += 1
            # transfer customer from checkout queue to processing queue
            _, _, proc_customer = heapq.heappop(checkout.queue)
            heapq.heappush(
                checkout.processing,
                (proc_customer.t_dep, proc_customer.cust_id, proc_customer),
            )
            # check if capacity is reached and if so, change status
            if checkout.c_type == "cc":
                if checkout.proc_num == checkout.c_quant:
                    checkout.c_status = 1
            elif checkout.c_type == "sc":
                if checkout.proc_num == checkout.sc_quant:
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
        with tqdm(total=self.t_max, unit_scale=True) as pbar:
            while self.t < self.t_max:
                t_old = float(self.t)
                self.next_action()
                t_delta = float(self.t) - t_old
                pbar.update(t_delta)

        # TODO: hier data wrangling einfügen, damit logs direkt für Datenanalyse genutzt werden können
        event_df = pd.DataFrame(self.event_log)
        event_df.columns = ["event_id", "time", "kind", "customer_id", "checkout_id"]

        customer_df = pd.DataFrame(self.customer_log)
        customer_df.columns = ["customer_id", "arrival_time", "departure_time", "processing_rate"]

        queue_df = pd.DataFrame(self.queue_log)
        queue_df.columns = [
            "time",
            "c_ql",
        ]
        columns = [key for key in self.checkouts.keys()]
        queue_df[columns] = pd.DataFrame(queue_df.c_ql.to_list(), index=queue_df.index)
        queue_df.drop("c_ql", inplace=True, axis=1)

        return event_df, customer_df, queue_df


def main():
    my_sim = Simulation(
        num_cc=16,
        num_sc=6,
    )
    event_log, customer_log, queue_log = my_sim.simulate()

    event_log.to_csv("event_log1.csv", index=False)
    customer_log.to_csv("customer_log1.csv", index=False)
    queue_log.to_csv("queue_log1.csv", index=False)


# signalize to reader of code that this is a script and not just a library
if __name__ == "__main__":
    main()
