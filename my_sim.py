import heapq
import warnings
# use dataclasses to improve readability
from dataclasses import dataclass, field
from itertools import count
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# ignore Warning, that the progress bar might go slightly over 100% due to rounding errors
warnings.filterwarnings("ignore", module="tqdm")

# TODO: Verteilungen für Service Time abstrahieren
"""
Funktionen als Parameter einbauen und dann normal (?) und exponentialverteilung 
zusätzlich zu denen aus POS Daten einbauen
"""
# TODO: Codefragmente die Duplikate sind zusammenführen

# TODO: Experiment Design festlegen (einfach unterschiedliche Parameter nutzen und dann Plots machen und vergleichen)
"""
Experiment Design: 
- POS Daten nutzen
- Verteilungen herausfinden
- Systemerweiterungen analysieren (Cashier Checkouts hinzufügen, Self Checkouts hinzufügen)
Das muss genug sein! Normal und Exponentialverteilung werden eingebaut, aber werde ich nicht nutzen
"""
# TODO: Visualisierungen erstellen
"""
hier dann einfach Notebook einfügen und auch mit Visualisierungen zu unterschiedlichen Systemen schmücken
"""


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
    t_arr: float = None
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
    c_status: int = 0
    """ status of the cashier/self-checkout {0: free, 1: Busy}"""
    c_quant: int = 1
    """ number of cashiers """
    sc_quant: int = 6
    """ number of self checkouts """
    queue: List[Tuple[float, int, Customer]] = field(default_factory=list)
    """ list of costumers in queue """
    processing: List[Tuple[float, int, Customer]] = field(default_factory=list)
    """ list of customers in processing """

    # heapify the queue and processing list and make sure c_status is always 0,
    def __post_init__(self):
        heapq.heapify(self.queue)
        heapq.heapify(self.processing)


class Simulation:
    """Class for simulation of a supermarket"""

    def __init__(
            self,
            s_seed: int = 42,
            t_max: float = 1000,
            # TODO: Hier unterschiedliche processing Verteilungen einfügen?
            distribution: str = "POS",
            proc_exp_cc: float = 2.5,
            proc_exp_sc: float = 0.75,
            proc_pos_cc_loc: float = 3.7777777777777777,
            proc_pos_cc_scale: float = 2.1742325579116906,
            proc_pos_sc_loc: float = 11.224246069610276,
            proc_pos_sc_scale: float = 6.208811868891992,
            arrival_rate: float = 1.0,
            num_cc: int = 6,
            num_sc: int = 1,
            c_quant: int = 1,
            sc_quant: int = 6,
            item_scale: float = 14.528563291255535,  # Scale parameter for exponential distribution
    ):
        """initialize parameters"""
        self.processing_parameters_exp = {"cc": proc_exp_cc, "sc": proc_exp_sc}
        self.processing_parameters_pos = {
            "cc_loc": proc_pos_cc_loc,
            "cc_scale": proc_pos_cc_scale,
            "sc_loc": proc_pos_sc_loc,
            "sc_scale": proc_pos_sc_scale,
        }
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
        self.distribution_choice = distribution

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

        """If queuing mode is shortest:
        sample arrival time, choose the shortest queue, generate a new_customer and add the new arrival event to the
        event list of the class object
        """
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
        """update the list of queue lengths and write it to the queue log"""
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
            if self.distribution_choice == "POS":
                if checkout.c_type == "cc":
                    proc_rate_per_item = self.rng.laplace(
                        loc=self.processing_parameters_pos["cc_loc"],
                        scale=self.processing_parameters_pos["cc_scale"],
                    )
                elif checkout.c_type == "sc":
                    proc_rate_per_item = self.rng.gumbel(
                        loc=self.processing_parameters_pos["sc_loc"],
                        scale=self.processing_parameters_pos["sc_scale"],
                    )
                else:
                    raise ValueError('Checkout Type is neither "cc" nor "sc"')
            elif self.distribution_choice == "exp":
                if checkout.c_type == "cc":
                    proc_rate_per_item = self.rng.exponential(
                        self.processing_parameters_exp["cc"]
                    )
                elif checkout.c_type == "sc":
                    proc_rate_per_item = self.rng.exponential(
                        self.processing_parameters_exp["sc"]
                    )
                else:
                    raise ValueError('Checkout Type is neither "cc" nor "sc"')
            else:
                raise NotImplementedError
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
            # check if capacity of cashier is reached and set status to busy
            if checkout.c_type == "cc":
                if len(checkout.processing) == checkout.c_quant:
                    checkout.c_status = 1
            elif checkout.c_type == "sc":
                if len(checkout.processing) == checkout.sc_quant:
                    checkout.c_status = 1

        # generate new arrival event
        self.get_arrival()

    def departure(self):
        """
        Pops the departure event from the event list and logs departure.

        If the queue is empty a new customer is popped from the queue and enters the processing queue.
        Processing rate per item, total processing rate and departure time are calculated. A new departure event is
        pushed into the event list.
        """
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
        heapq.heappop(checkout.processing)

        if len(checkout.queue) > 0:
            # generate random processing time
            # legacy code: proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            if self.distribution_choice == "POS":
                if checkout.c_type == "cc":
                    proc_rate_per_item = self.rng.laplace(
                        loc=self.processing_parameters_pos["cc_loc"],
                        scale=self.processing_parameters_pos["cc_scale"],
                    )
                elif checkout.c_type == "sc":
                    proc_rate_per_item = self.rng.gumbel(
                        loc=self.processing_parameters_pos["cc_loc"],
                        scale=self.processing_parameters_pos["cc_scale"],
                    )
                else:
                    raise ValueError('Checkout Type is neither "cc" nor "sc"')

            elif self.distribution_choice == "exp":
                if checkout.c_type == "cc":
                    proc_rate_per_item = self.rng.exponential(
                        self.processing_parameters_exp["cc"]
                    )
                elif checkout.c_type == "sc":
                    proc_rate_per_item = self.rng.exponential(
                        self.processing_parameters_exp["sc"]
                    )
                else:
                    raise ValueError('Checkout Type is neither "cc" nor "sc"')

            else:
                raise NotImplementedError
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
            # transfer customer from checkout queue to processing queue
            _, _, proc_customer = heapq.heappop(checkout.queue)
            heapq.heappush(
                checkout.processing,
                (proc_customer.t_dep, proc_customer.cust_id, proc_customer),
            )
            # check if capacity is reached and if so, change status
            if checkout.c_type == "cc":
                if len(checkout.processing) == checkout.c_quant:
                    checkout.c_status = 1
            elif checkout.c_type == "sc":
                if len(checkout.processing) == checkout.sc_quant:
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

        event_df = pd.DataFrame(self.event_log)
        event_df.columns = ["event_id", "time", "kind", "customer_id", "checkout_id"]

        customer_df = pd.DataFrame(self.customer_log)
        customer_df.columns = [
            "customer_id",
            "arrival_time",
            "departure_time",
            "processing_rate",
        ]

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
        distribution='exp',
    )
    event_log, customer_log, queue_log = my_sim.simulate()

    event_log.to_csv("event_log1.csv", index=False)
    customer_log.to_csv("customer_log1.csv", index=False)
    queue_log.to_csv("queue_log1.csv", index=False)


# signalize to reader of code that this is a script and not just a library
if __name__ == "__main__":
    main()
