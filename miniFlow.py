
# coding: utf-8

# # MiniFlow Architecture
# small tensorFlow clone

# ## Generic node representation class
# Each node might receive input from multiple other nodes.
# each node creates a single output, which will likely be passed to other nodes.
# 
# Contains two lists:
# - one to store references to the inbound ndoes.
# - one to store references to the outbound nodes.
# 
# Each node will eventually calculate a value that represents its output.
# - initialize the value to None to indicate that it exists but hasn't been set yet.
# 
# Each node will need to be able to pass values forward and perform backpropagation.
# ...

# In[1]:

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Properties
        
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # For each inbound Node here, add this NOde as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        
        
        def forward(self):
            """
            Forward propagation.
            
            Compute the output value vased on `inbound_nodes` and store the result in self.value.
            """
            raise NotImplemented


# ## Nodes that Calculate - subclasses
# While `Node` defines the base set of properties that every node holds, only specialized [subclasses](https://docs.python.org/3/tutorial/classes.html#inheritance) of `Node` will end up in the graph.
# 
# The following subclasses of `Node` will be created:
# - `Input`
# - `Add`

# ### The `Input` Subclass

# In[2]:

class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)
        
    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value


# Unlike the other subclasses of `Node`, the `Input` subclass does not actually calculate anything. The `Input` subclass just holds a `value`, such as a data feature or a model parameter (weight/bias).
# 
# The `value` can be either set explicitly or with the `forward()` method. This value is then fed through the rest of the neural network.

# ### The `Add` Subclass

# `Add`, which is another subclass of `Node`, actually can perform a calculation (addition).

# In[3]:

class Add(Node):
    def __init__(self, x, y):
        # Pass the inbound_nodes list [x, y] to 
        # the Node instantiator. The Add subclass takes two
        # inbound nodes, x and y, and adds the values of those nodes.
        Node.__init__(self, [x, y]) 
        
    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of it's inbound_nodes.
        
        """
        self.value = self.inbound_nodes[0].value + self.inbound_nodes[1].value


# ## Forward propagation
# 
# `MiniFlow` has two methods to help define and then run values through graphs:
# - `topological_sort()`
# - `forward_pass()`
# 
# The picture from Udacity shows an example of topological sorting

# ![An example of top](./topological-sort001.jpeg)

# In order to define a network, we need to define the order of operations for nodes. Given that the input to some node depends on the outputs of others, we need to flatten the graph in such a way where all the input dependencies for each node are resolved before trying to run its calculation. This is a technique called [topological sort](https://en.wikipedia.org/wiki/Topological_sorting)
# 
# The `topological_sort()` function implements topological sorting using [Kahn's Algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm). 
# 
# `topological_sort()` returns a sorted list of nodes in which all of the calculations can run in series.
# 
# topological_sort() takes in a feed_dict, which is how we initially set a value for an Input node. The feed_dict is represented by the Python dictionary data structure.

# In[4]:

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


# `forward_pass()` actually runs the network and outputs a value.

# In[5]:

def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value


# In[6]:

def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value

