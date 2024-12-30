class BoundedList:
    """
    list with bounded size.
    if new element is pushed and max_size is reached, first element would be overwritten by a new element
    """
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.items = []

    def add(self, value):
        if len(self.items) >= self.max_size:
            # Remove the first element if the list is at max capacity
            self.items.pop(0)
        # Add the new value to the end of the list
        self.items.append(value)

    @property
    def unique_length(self):
        return len(set(self.items))

    def __repr__(self):
        return f"{self.items}"
