"""
Simple asynchronous event bus for inter-module communication.
"""
from collections import defaultdict

class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event_type, callback):
        self.subscribers[event_type].append(callback)

    def publish(self, event_type, data):
        for callback in self.subscribers[event_type]:
            callback(data)
