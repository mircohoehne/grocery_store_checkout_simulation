import heapq
import warnings

from dataclasses import dataclass, field
from itertools import count
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# ignore Warning, that the progress bar might go slightly over 100% due to rounding errors
warnings.filterwarnings("ignore", module="tqdm")


@dataclass(slots=True, frozen=True)
class Event:
    """Represents a discrete event in the supermarket simulation.

    Events are immutable and ordered by their occurrence time. Used to drive
    the discrete event simulation forward.

    Attributes:
        ev_id: Unique event identifier (auto-generated)
        time: Simulation time when event occurs (in seconds)
        kind: Type of event ('arr' for arrival, 'dep' for departure)
        customer: Customer object associated with the event
        c_id: Checkout station ID where event occurs (1-based index)
    """

    ev_id: int = field(default_factory=count(start=1).__next__, init=False)
    time: float
    kind: str
    customer: object
    c_id: int


@dataclass(slots=True)
class Customer:
    """Represents a customer moving through the supermarket system.

    Tracks key timestamps and processing parameters for an individual customer's journey.

    Attributes:
        cust_id: Unique customer identifier (auto-generated)
        t_arr: Arrival time at checkout area (in seconds)
        t_dep: Departure time from checkout (in seconds)
        t_proc: Total processing time required (in seconds)
        num_items: Number of items in shopping basket
        proc_rate_per_item: Processing time per item (in seconds/item)
        c_id: Checkout station ID assigned to (1-based index)
    """

    cust_id: int = field(default_factory=count(start=1).__next__, init=False)
    t_arr: float = None
    t_dep: float = None
    t_proc: float = None
    num_items: int = None
    proc_rate_per_item: float = None
    c_id: int = None


@dataclass(slots=True)
class Checkout:
    """Represents a checkout station in the supermarket.

    Manages queue state and processing capacity for either cashier-operated or
    self-service checkout stations.

    Attributes:
        c_id: Unique station identifier (1-based index)
        c_type: Station type ('cc' for cashier, 'sc' for self-checkout)
        c_status: Operational status (0: available, 1: at full capacity)
        c_quant: Number of cashier stations
        sc_quant: Number of self-checkout stations
        queue: Priority queue of waiting customers (sorted by arrival time)
        processing: Priority queue of active customers (sorted by departure time)

    Note:
        Uses heapq internally to maintain efficient queue operations
    """

    c_id: int
    c_type: str
    c_status: int = 0
    c_quant: int = 1
    sc_quant: int = 6
    queue: List[Tuple[float, int, Customer]] = field(default_factory=list)
    processing: List[Tuple[float, int, Customer]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize heap structures for queue and processing lists."""
        heapq.heapify(self.queue)
        heapq.heapify(self.processing)


class Simulation:
    """Class for simulation of a supermarket checkout system.

    Simulates customer arrivals, queue management, and checkout processing using discrete event simulation.
    Tracks system state through event logging, customer flow, and queue lengths over time.

    Args:
        s_seed: Random number generator seed
        t_max: Simulation time in seconds
        distribution: Processing time distribution ("POS" for distribution following the POS Data or "exp" for exponential distribution)
        proc_exp_cc: Exponential distribution parameter for cashier checkouts
        proc_exp_sc: Exponential distribution parameter for self-checkouts
        proc_pos_cc_loc: POS data location parameter for transaction time on cashier checkouts
        proc_pos_cc_scale: POS data scale parameter for transaction time on cashier checkouts
        proc_pos_sc_loc: POS data location parameter for transaction time on self-checkouts
        proc_pos_sc_scale: POS data scale parameter for transaction time on self-checkouts
        arrival_rate: Arrival rate of customers
        num_cc: Number of cashier-operated checkouts
        num_sc: Number of self-service checkouts
        c_quant: Number of customers that can be handled at once at cashier checkouts
        sc_quant: Number of customers that can be handled at once at self-checkouts
        item_scale: Scale parameter for the number of items a customer buys

    Attributes:
        event_log: List of all simulated events
        customer_log: List of customer journey records
        queue_log: Historical record of queue lengths
        checkouts: Dictionary of checkout stations
    """

    def __init__(
        self,
        s_seed: int = 42,
        t_max: float = 1000,
        distribution: str = "POS",
        proc_exp_cc: float = 2.5,
        proc_exp_sc: float = 0.75,
        proc_pos_cc_loc: float = 3.7777777777777777,
        proc_pos_cc_scale: float = 2.1742325579116906,
        proc_pos_sc_loc: float = 11.224246069610276,
        proc_pos_sc_scale: float = 6.208811868891992,
        arrival_rate: float = 3.5,
        num_cc: int = 6,
        num_sc: int = 1,
        c_quant: int = 1,
        sc_quant: int = 6,
        item_scale: float = 14.528563291255535,
    ):
        """Initialize simulation parameters and system state.

        Creates checkout stations, initializes logging structures,
        and schedules first arrival event.
        """
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
        """Schedule next customer arrival and assign to shortest queue.

        Generates:
            - Inter-arrival time using exponential distribution
            - Customer's item count using exponential distribution
            - New arrival event added to event queue
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
        # set checkout id
        new_customer.c_id = c_id
        # create new event
        new_arr = Event(t_arrival, "arr", new_customer, c_id)
        # push event into heapq, use ev_id for sorting if events occur at same time
        # and c_id to schedule new events
        heapq.heappush(self.event_list, (t_arrival, new_arr.ev_id, new_arr))

    def update_ql(self):
        """update the list of queue lengths and write it to the queue log"""
        new_ql = []
        for i in range(self.sum_c):
            new_ql.append(len(self.checkouts[f"Checkout{i + 1}"].queue))
        self.queue_log.append((self.t, [new_ql[x] for x in range(self.sum_c)]))
        self.ql_list = new_ql

    def arrival(self):
        """Process customer arrival event.

        Steps:
            1. Remove arrival event from queue
            2. Add customer to selected checkout queue
            3. If checkout available, start processing
            4. Schedule next arrival event
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
            # generate processing time per item while the processing rate per item is none or <= 0
            while (
                proc_customer.proc_rate_per_item == None
                or proc_customer.proc_rate_per_item <= 0
            ):
                if self.distribution_choice == "POS":
                    if checkout.c_type == "cc":
                        proc_customer.proc_rate_per_item = self.rng.laplace(
                            loc=self.processing_parameters_pos["cc_loc"],
                            scale=self.processing_parameters_pos["cc_scale"],
                        )
                    elif checkout.c_type == "sc":
                        proc_customer.proc_rate_per_item = self.rng.gumbel(
                            loc=self.processing_parameters_pos["sc_loc"],
                            scale=self.processing_parameters_pos["sc_scale"],
                        )
                    else:
                        raise ValueError('Checkout Type is neither "cc" nor "sc"')
                elif self.distribution_choice == "exp":
                    if checkout.c_type == "cc":
                        proc_customer.proc_rate_per_item = self.rng.exponential(
                            self.processing_parameters_exp["cc"]
                        )
                    elif checkout.c_type == "sc":
                        proc_customer.proc_rate_per_item = self.rng.exponential(
                            self.processing_parameters_exp["sc"]
                        )
                    else:
                        raise ValueError('Checkout Type is neither "cc" nor "sc"')
                else:
                    raise NotImplementedError
            # calculate processing time for all items
            proc_rate = proc_customer.proc_rate_per_item * proc_customer.num_items
            # set proc rate for customer
            proc_customer.t_proc = proc_rate
            # calculate departure time
            t_1 = self.t + proc_rate
            # create new event
            new_dep = Event(t_1, "dep", proc_customer, proc_customer.c_id)
            # add new departure event
            heapq.heappush(self.event_list, (t_1, new_dep.ev_id, new_dep))
            # add departure and processing time for customer
            proc_customer.t_dep = t_1
            # put customer in processing queue
            heapq.heappush(
                checkout.processing,
                (proc_customer.t_dep, proc_customer.cust_id, proc_customer),
            )
            # check if capacity of cashier is reached and if so, change status
            if checkout.c_type == "cc":
                if len(checkout.processing) == checkout.c_quant:
                    checkout.c_status = 1
            elif checkout.c_type == "sc":
                if len(checkout.processing) == checkout.sc_quant:
                    checkout.c_status = 1

        # generate new arrival event
        self.get_arrival()

    def departure(self):
        """Process customer departure event.

        Steps:
            1. Remove departure event from queue
            2. Free up checkout capacity
            3. If customers waiting, start next processing
            4. Log customer journey statistics
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
            # get customer from checkout queue
            dep_customer = checkout.queue[0][2]
            # generate random processing time
            # legacy code: proc_rate = self.rng.exponential(self.processing_rate[checkout.c_type])
            while (
                dep_customer.proc_rate_per_item == None
                or dep_customer.proc_rate_per_item <= 0
            ):
                if self.distribution_choice == "POS":
                    if checkout.c_type == "cc":
                        dep_customer.proc_rate_per_item = self.rng.laplace(
                            loc=self.processing_parameters_pos["cc_loc"],
                            scale=self.processing_parameters_pos["cc_scale"],
                        )
                    elif checkout.c_type == "sc":
                        dep_customer.proc_rate_per_item = self.rng.gumbel(
                            loc=self.processing_parameters_pos["sc_loc"],
                            scale=self.processing_parameters_pos["sc_scale"],
                        )
                    else:
                        raise ValueError('Checkout Type is neither "cc" nor "sc"')

                elif self.distribution_choice == "exp":
                    if checkout.c_type == "cc":
                        dep_customer.proc_rate_per_item = self.rng.exponential(
                            self.processing_parameters_exp["cc"]
                        )
                    elif checkout.c_type == "sc":
                        dep_customer.proc_rate_per_item = self.rng.exponential(
                            self.processing_parameters_exp["sc"]
                        )
                    else:
                        raise ValueError('Checkout Type is neither "cc" nor "sc"')

                else:
                    raise NotImplementedError
            # calculate total proc_rate for all items
            proc_rate = dep_customer.num_items * dep_customer.proc_rate_per_item
            # save proc_rate for departing customer
            dep_customer.t_proc = proc_rate
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
            # check if capacity of cashier is reached and if so, change status
            if checkout.c_type == "cc":
                if len(checkout.processing) == checkout.c_quant:
                    checkout.c_status = 1
            elif checkout.c_type == "sc":
                if len(checkout.processing) == checkout.sc_quant:
                    checkout.c_status = 1

        # log customer
        total_time = current_event.customer.t_dep - current_event.customer.t_arr
        self.customer_log.append(
            [
                current_event.customer.cust_id,
                current_event.customer.t_arr,
                current_event.customer.t_dep,
                current_event.customer.t_proc,
                total_time,
                current_event.customer.c_id,
            ]
        )

    def next_action(self):
        if self.event_list[0][2].kind == "arr":
            self.arrival()
        elif self.event_list[0][2].kind == "dep":
            self.departure()

    def simulate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the simulation until time limit is reached.

        Returns:
            Tuple containing:
                - event_df: DataFrame of all events
                - customer_df: DataFrame of customer journeys
                - queue_df: DataFrame of queue lengths over time

        Note:
            Uses tqdm progress bar to display simulation progress
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
            "time from queuing to checkout",
            "checkout id",
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


if __name__ == "__main__":
    """
    Experiment 1:
    """
    my_sim = Simulation(
        num_cc=16,
        num_sc=1,
        t_max=10000,  # six self checkouts with one queue
    )
    event_log, customer_log, queue_log = my_sim.simulate()

    event_log.to_csv("event_log1.csv", index=False)
    customer_log.to_csv("customer_log1.csv", index=False)
    queue_log.to_csv("queue_log1.csv", index=False)

    """
    Experiment 2:
    """
    my_sim = Simulation(
        num_cc=22,
        num_sc=1,
        t_max=10000,  # six self checkouts with one queue
    )
    event_log, customer_log, queue_log = my_sim.simulate()

    event_log.to_csv("event_log2.csv", index=False)
    customer_log.to_csv("customer_log2.csv", index=False)
    queue_log.to_csv("queue_log2.csv", index=False)

    """
    Experiment 3:
    """
    my_sim = Simulation(
        num_cc=16,
        num_sc=2,
        t_max=10000,  # six self checkouts with one queue
    )
    event_log, customer_log, queue_log = my_sim.simulate()

    event_log.to_csv("event_log3.csv", index=False)
    customer_log.to_csv("customer_log3.csv", index=False)
    queue_log.to_csv("queue_log3.csv", index=False)
