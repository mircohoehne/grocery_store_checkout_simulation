import numpy as np
import heapq
from itertools import count

# use dataclasses to improve readability
from dataclasses import dataclass, field


# initialized with slots to increase performance and frozen to prevent changing of values
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
    def get_event_data(self) -> (float, str, int):
        """
        :return: return event data for the event list
        """
        return self.time, self.kind, self.c_id


# welchen Datentyp hat eigentlich Zeit? Int? -> erstmal float nutzen
# Everything is stored in one event list!
# Generate Arrival events
# wirklich notwendig rng zu übergeben oder reicht dann numpy dependency?
def get_arrival(event_list: list, rng: np.random._generator.Generator, t: float):
    t_arrival = t + rng.exponential()
    raise NotImplementedError


@dataclass(slots=True, frozen=True)
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
# so kann das einfach vom arrival prozess abgefragt werden!!!
# queue length als Liste für alle Kassen anlegen, so kann min gesucht
# werden!


def simulation(
    num_cc: int,
    num_sc: int,
    t_max: float,
    run_num: int,
) -> None:

    # initialize rng generator
    rng = np.random.default_rng(seed=42)
    # Initialize Event list
    eventlist = []
    heapq.heapify(eventlist)
    # Initialize time
    t = 0.0
    # initialize queue length dictionary
    checkouts = {}

    # initialize cashier and self-checkouts and store the checkout objects in a dictionary
    for i in range(num_cc):
        key = f"Checkout{i+1}"
        id = i + 1
        checkouts[key] = Checkout((i + 1), "cc")
    for j in range(num_cc, num_cc + num_sc):
        key = f"Checkout{j+1}"
        id = j + 1
        checkouts[key] = Checkout((j + 1), "sc")

    print(checkouts)
    # method to advance time
    while t < t_max:
        t = t + 1
        pass

    # print loop for testing purposes
    for i in range(num_cc + num_sc):
        print(checkouts[f"Checkout{i+1}"].ql)


simulation(6, 2, 5, 4)
