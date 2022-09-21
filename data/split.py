'''
This file is to split Coauthor/Amazon dataset
'''

import numpy as np

def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def get_split(labels, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, test_size=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class, forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    # if train_examples_per_class is not None:
    #     train_labels = labels[train_indices, :]
    #     train_sum = np.sum(train_labels, axis=0)
    #     # assert all classes have equal cardinality
    #     assert np.unique(train_sum).size == 1
    #
    # if val_examples_per_class is not None:
    #     val_labels = labels[val_indices, :]
    #     val_sum = np.sum(val_labels, axis=0)
    #     # assert all classes have equal cardinality
    #     assert np.unique(val_sum).size == 1
    #
    # if test_examples_per_class is not None:
    #     test_labels = labels[test_indices, :]
    #     test_sum = np.sum(test_labels, axis=0)
    #     # assert all classes have equal cardinality
    #     assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices