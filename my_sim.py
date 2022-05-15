import heapq
# use dataclasses to improve readability
from dataclasses import dataclass, field
from itertools import count
from typing import Tuple, Union

import numpy as np


# initialized with slots to increase performance and frozen to prevent changing of values
# muss frozen auf false gesetzt werden um departure Zeit zu erfassen?
@dataclass(slots=True, frozen=True)
class Customer:
    """class to keep track of customers"""

    # initialize Variables for Object
    t_arr: float
    """arrival time of the customer"""
    cust_id: int = field(default_factory=count().__next__, init=False)
    """generate an unique id for every customer"""


# Überhaupt möglich mit heap event klasse zu speichern? muss sonst einfach die Zeit vor event objekt stehen?
@dataclass(slots=True, frozen=True)
class Event:
    """class for keeping track of events"""

    # initialize variables for object
    time: float
    """time at which event occurs"""
    kind: str
    """kind of event "arr": arrival and "dep": departure"""
    customer: object
    """customer that is handled"""
    c_id: int
    """id of checkout where event occurs"""
    # init is set to false, so the counter doesn't reset every time
    ev_id: int = field(default_factory=count().__next__, init=False)
    """generate a new id when a class object is created, used for sorting purposes in heapq"""

    # define methods
    # unnötige methode?
    def get_event_data(self) -> Tuple[float, str, int]:
        """
        :return: return event data for the event list
        """
        return self.time, self.kind, self.c_id


# welchen Datentyp hat eigentlich Zeit? Int? → erstmal float nutzen
# Everything is stored in one event list!
# Generate Arrival events
# wirklich notwendig rng zu übergeben oder reicht dann numpy dependency?


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


# use heapq.heapify(list) to make list into heap and use heappush to insert elements
# use time as the first value in tuple, this is the value that is sorted by!


# queue length als counter bei jeder einzelnen kasse implementieren,
# so kann das einfach vom arrival prozess abgefragt werden!
# queue length als Liste für alle Kassen anlegen, so kann min gesucht
# werden!

################### Static Methods ####################
def get_arrival(
        event_list: list[Union[int, Event]], rng: np.random.Generator, t: float
):
    t_arrival = t + rng.exponential()
    raise NotImplementedError


def get_ql(sum_c: int, checkouts: dict) -> list[int]:
    ql_list = []
    for i in range(sum_c):
        ql_list.append(checkouts[f"Checkout{i + 1}"].ql)
    return ql_list


###################### Main Function #####################
def simulation(
        num_cc: int,
        num_sc: int,
        t_max: float,
        # define number of simulations in function or create a for loop outside?
        run_num: int = 1,
) -> None:
    # store nr. of checkouts
    sum_c = num_cc + num_sc
    # initialize rng generator
    rng = np.random.default_rng(seed=42)
    # Initialize Event list
    event_list: list[Union[int, Event]] = []
    heapq.heapify(event_list)
    # Initialize time
    t = 0.0
    # initialize dictionary that holds the Checkout object
    checkouts = {}

    # initialize cashier and self-checkouts and store the checkout objects in a dictionary
    for i in range(num_cc):
        num = i + 1
        key = f"Checkout{num}"
        checkouts[key] = Checkout(num, "cc")
    for j in range(num_cc, sum_c):
        num = j + 1
        key = f"Checkout{num}"
        checkouts[key] = Checkout(num, "sc")

    # method to advance time
    while t < t_max:
        t = t + 1
        pass

    # print loop for testing purposes
    for i in range(num_cc + num_sc):
        print("------------------------------")
        print(f"Checkout{i + 1} = {checkouts[f'Checkout{i + 1}']}")

    my_list = get_ql(sum_c, checkouts)
    print(my_list)


simulation(3, 1, 100)
