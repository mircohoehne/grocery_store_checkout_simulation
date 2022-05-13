import numpy as np
import heapq
import itertools


class Customer:
    # class for keeping track of customers
    # generate a new id for each customer
    cust_id = itertools.count()

    def __init__(self, time: float):
        self.custom_id = next(Customer.cust_id)
        self.arrival_time = time

    def __str__(self):
        return (
            f"The customer has id: {self.custom_id} and arrived at: {self.arrival_time}"
        )

    # id is added to repr just for debugging purposes
    def __repr__(self):
        return f"Customer(arrival={self.arrival_time}, id={self.custom_id})"


# Überhaupt möglich mit heap event klasse zu speichern? muss sonst einfach die Zeit vor event objekt stehen?
class Event:
    # class for keeping track of events
    # generate an id for every event for sorting purposes in heapq
    ev_id = itertools.count()

    def __init__(self, time: float, kind: str, c_id: int, customer: object) -> None:
        """

        :param time: time at which the event occurs
        :param kind: defines the kind of event "arr" is arrival and "dep" is departure
        :param c_id: id of the Checkout at which the event occurs
        """
        self.time = time
        self.kind = kind
        self.c_num = c_id

    def __str__(self):
        return self.kind

    def __repr__(self):
        return f'Event(time={self.time}, kind="{self.kind}", id={self.c_num})'

    # unnötige Funktion?
    def get_event_data(self) -> (float, str, int):
        """
        :return: return event data for the event list
        """
        return self.time, self.kind, self.c_num


# welchen Datentyp hat eigentlich Zeit? Int? -> erstmal float nutzen
# Everything is stored in one event list!
# Generate Arrival events
# wirklich notwendig rng zu übergeben oder reicht dann numpy dependency?
def get_arrival(event_list: list, rng: np.random._generator.Generator, t: float):
    t_arrival = t + rng.exponential()
    raise NotImplementedError


# Sollte Checkout noch eine Liste übergeben bekommen in der Länger der queue geloggt
# wird zum Auswählen der Kasse bei Ankunft?
class Checkout:
    def __init__(self, c_id: int, c_type: str) -> None:
        """
        :param c_id: id of the checkout
        :param c_type: type of the checkout
        """
        self.c_id = c_id
        self.c_type = c_type
        # initialize cashier_status and queue length to zero
        self.c_status = 0
        self.ql = 0

    def __str__(self):
        return self.c_type

    def __repr__(self):
        return f'Checkout(id={self.c_id}, type="{self.c_type}")'


# use heapq.heapify(list) to make list into heap and use heappush to insert elements
# use time as the first value in tuple, this is the value that is sorted by!


# queue length als counter bei jeder einzelnen kasse implementieren,
# so kann das einfach vom arrival prozess abgefragt werden!!!
# queue length als Liste für alle Kassen anlegen, so kann min gesucht
# werden!


def simulation(num_cc: int, num_sc: int, t_max, run_num) -> None:
    # initialize rng generator
    rng = np.random.default_rng(seed=42)
    # Initialize Event list
    eventlist = []
    heapq.heapify(eventlist)
    # Initialize time
    t = 0.0
    # initialize queue length dictionary
    checkouts = {}

    # initialize cashier and self checkouts and store the checkout objects in a dictionary
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
