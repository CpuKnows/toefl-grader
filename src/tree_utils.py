import re


class Node(object):
    def __init__(self, node_id, label, parent=None, children=[], word=None):
        """Creates a tree node"""
        assert len(children) == 0 or word is None, 'Can only have children nodes or a word'
        if parent is not None:
            assert type(parent) == Node, 'Parent must be a Node object'
        if children is not None:
            for child in children:
                assert type(child) == Node, 'All children must be Node objects'
                
        self.node_id = node_id
        self.label = label
        self.parent = parent
        self.children = children
        self.word = word
        
    def __str__(self):
        if self.parent is None:
            return ('Node id:{} label:{} word:{} parent_id:{} num_children:{}'.format(
                self.node_id, self.label, self.word, None, len(self.children)
            ))
        else:
            return ('Node id:{} label:{} word:{} parent_id:{} num_children:{}'.format(
                self.node_id, self.label, self.word, self.parent.node_id, len(self.children)
            ))

    def __contains__(self, label):
        """Is PTB tag in node/children labels"""
        if label == self.label:
            return True
        for child in self.children:
            if label in child:
                return True
        return False
        
    def is_production(self):
        return True if self.word is None else False
    
    def is_leaf(self):
        return True if self.word is not None else False

    def has_parent(self):
        return True if self.parent is not None else False

    def has_children(self):
        return True if len(self.children) > 0 else False
        
    def get_siblings(self):
        siblings = []
        if self.has_parent():
            for child in self.parent.children:
                if child.node_id != self.node_id:
                    siblings.append(child)
        return siblings
        
    def get_left_siblings(self):
        siblings = []
        if self.has_parent():
            for child in self.parent.children:
                if child.node_id < self.node_id:
                    siblings.append(child)
        return siblings
        
    def get_right_siblings(self):
        siblings = []
        if self.has_parent():
            for child in self.parent.children:
                if child.node_id > self.node_id:
                    siblings.append(child)
        return siblings
    
    def add_child(self, child):
        self.children.append(child)

    def get_ancestors(self, label=None):
        if self.has_parent():
            if label is None:
                yield self.parent
            elif self.parent.label == label:
                yield self.parent
            yield from self.parent.get_ancestors( label)

    def get_descendants(self, label=None):
        for child in self.children:
            if label is None:
                yield child
            elif label == child.label:
                yield child
            yield from child.get_descendants(label)


def rightmost_leaf_id(node):
    current_node = node
    while len(current_node.children) != 0:
        current_node = current_node.children[-1]
    return current_node.node_id


def parse_tree_regex(tree_str):
    # Grammar rule
    rule_match = re.match('\([\S]+', tree_str)
    if rule_match is not None:
        rule_match = rule_match.group()
        tree_str = tree_str[len(rule_match):].strip()
        rule_match = rule_match.lstrip('(')

        # Leaf node
        word_match = re.match('[\S]+\)', tree_str)
        if word_match is not None:
            word_match = word_match.group()
            tree_str = tree_str[len(word_match):].strip()
            word_match = word_match.rstrip(')')
                
    return rule_match, word_match, tree_str


def create_tree(tree_str, parent=None):
    def _create_tree(tree_str, parent=None):
        """Recursively build tree data structure from parsed sentence"""
        tree_str = tree_str.strip()
        rule_match = None
        word_match = None
        
        if parent is None:
            # Root node
            rule_match, word_match, tree_str = parse_tree_regex(tree_str)
            # Create new node
            new_id = 0
            new_node = Node(new_id, rule_match, parent=parent, children=[], word=word_match)
            return _create_tree(tree_str, new_node)
        
        # Non-root node
        while tree_str != '' and tree_str[0] != ')':
            rule_match, word_match, tree_str = parse_tree_regex(tree_str)
            # Create new node
            new_id = rightmost_leaf_id(parent) + 1
            new_node = Node(new_id, rule_match, parent=parent, children=[], word=word_match)
                
            if word_match is None:
                # Production rule
                tree_str, new_node = _create_tree(tree_str, new_node)
                parent.children.append(new_node)
            else:
                # leaf
                parent.children.append(new_node)
            
        if tree_str != '':
            return tree_str[1:], parent
        else:
            return tree_str, parent

    _, root_node = _create_tree(tree_str, parent)
    return root_node


def print_leaves(node):
    for child in node.children:
        if child.is_leaf():
            print(child.word, end=' ')
        else:
            print_leaves(child)
