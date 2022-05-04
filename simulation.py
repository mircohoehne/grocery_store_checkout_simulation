import numpy as np
import plotly.express as px
import pandas as pd


# create event-class
class event:
    # Übergebe Zeit und Art des Events
    # Art -> 0 ist Ankunft in Schlange, 1 ist verlassen der Kasse
    def __init__(self, time, kind):
        self.time = time
        self.kind = kind

        def __str__(self):
            return self.kind


# Definiere ein Ankunftsevent
def arrival(
    eventlist, cashier_status, queue_length, arrival_rate, processing_rate, t_0
):
    # erase current event from event list
    eventlist.remove(eventlist[0])
    # check if server is occupied
    if cashier_status == 0:
        # generate random processing time
        processing_rate = np.random.exponential(processing_rate)
        # add new departure event
        eventlist.append(event(t_0 + processing_rate, 1))
        # update cashier status -> busy
        cashier_status = 1
    else:
        # increase queue by 1
        queue_length += 1
    # generate new arrival event
    interarrival_time = np.random.exponential(arrival_rate)
    eventlist.append(event(t_0 + interarrival_time, 0))
    # sort eventlist -> complexity is O(n.log n),
    # maybe possible to make it more efficient with heap?
    eventlist.sort(key=lambda event: event.time)
    return eventlist, cashier_status, queue_length


# define departure event
def departure(eventlist, cashier_status, queue_length, processing_rate, t_0):
    # update cashier status
    cashier_status = 0
    # remove current event from event list
    eventlist.remove(eventlist[0])
    # check if events in queue
    if queue_length > 0:
        # generate random processing time
        processing_rate = np.random.exponential(processing_rate)
        # update event list
        eventlist.append(event(t_0 + processing_rate, 1))
        # update cashier status
        cashier_status = 1
        # decrease queue length
        queue_length -= 1
    # sort event list
    eventlist.sort(key=lambda event: event.time)
    return eventlist, cashier_status, queue_length


# define function for managing the event list
def next_action(
    eventlist, cashier_status, queue_length, arrival_rate, processing_rate, t_0
):
    t_0 = eventlist[0].time
    if eventlist[0].kind == 0:
        eventlist, cashier_status, queue_length = arrival(
            eventlist, cashier_status, queue_length, arrival_rate, processing_rate, t_0
        )
    elif eventlist[0].kind == 1:
        eventlist, cashier_status, queue_length = departure(
            eventlist, cashier_status, queue_length, processing_rate, t_0
        )
    return eventlist, queue_length, cashier_status, t_0


# define overall routine for simulation
def simulation(arrival_rate, processing_rate, t_max):
    eventlist = []
    cashier_status = 0
    queue_length = 0
    t_0 = 0

    ql_t = []
    t_1 = []
    # generate first arrival event
    interarrival_time = np.random.exponential(arrival_rate)
    eventlist.append(event(t_0 + interarrival_time, 0))

    # execute simulation until time limit
    while t_0 < t_max:
        eventlist, queue_length, cashier_status, t_0 = next_action(
            eventlist, cashier_status, queue_length, arrival_rate, processing_rate, t_0
        )
        # save queue length at state changes
        ql_t.append(queue_length)
        t_1.append(t_0)

    return ql_t, t_1


# SIM EXECUTION
arrival_rate = 5
processing_rate = 2
t_max = 1000

result = simulation(arrival_rate, processing_rate, t_max)


# visualization
# plt.plot(result[1], result[0])
# plt.ylabel("Queue Length")
# plt.show()

df = pd.DataFrame(list(zip(result[0], result[1])))
fig = px.line(df, x=1, y=0)
fig.show()
