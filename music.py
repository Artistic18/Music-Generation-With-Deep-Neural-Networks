# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:37:57 2020

@author: KIIT
"""

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils  import np_utils
from keras.callbacks import ModelCheckpoint

test = converter.parse('midi_songs/ahead_on_our_way_piano.mid')
parts = instrument.partitionByInstrument(test)
if parts:
    note = parts.parts[0].recurse()
else:
    note = test.flat.notes

for element in note:
    print(element)

def train_network():
    notes = get_data()
    n_vocab = len(set(notes))
    network_ip, network_op = input_sequence(notes, n_vocab)
    model = create_network(network_ip, n_vocab)
    train(model, network_ip, network_op)

def get_data():
    notes = []
    for song in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(song)
        print("Parsing %s" % song)
        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def input_sequence(notes,n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(i for i in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_ip = []
    network_op = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_ip.append([note_to_int[char] for char in sequence_in])
        network_op.append(note_to_int[sequence_out])
    n_patterns = len(network_ip)

    network_ip = numpy.reshape(network_ip, (n_patterns,sequence_length,1))
    network_ip = network_ip / float(n_vocab)
    network_op = np_utils.to_categorical(network_op)

    return (network_ip,network_op)

def create_network(network_ip, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_ip.shape[1], network_ip.shape[2]),
        recurrent_dropout = 0.3,
        return_sequences = True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_ip, network_op):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=True,
            save_best_only=True,
            mode='min'
            )
    callbacks_list = [checkpoint]
    model.fit(network_ip, network_op, epochs=300, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
   train_network()
