ALPcouplings
==============

Classes
-----------

.. autoclass:: alpaca.ALPcouplings
   :members:
   :exclude-members: from_dict, join_expressions, to_dict

   .. automethod:: alpaca.ALPcouplings.__init__


Objects
-----------

.. py:data:: alpaca.ALPcouplingsEncoder

   JSON encoder for ALPcouplings objects and structures containing them.

   Usage
   -----
   >>> import json
   >>> from alpaca import ALPcouplings, ALPcouplingsEncoder
   >>> a = ALPcouplings({'cG': 1.0}, 1e3, 'derivative_above')
   >>> with open('file.json', 'wt') as f:
   ...     json.dump(a, f, cls=ALPcouplingsEncoder)

.. py:data:: alpaca.ALPcouplingsDecoder

   JSON decoder for ALPcouplings objects and structures containing them.

   Usage
   -----
   >>> import json
   >>> from alpaca import ALPcouplingsDecoder
   >>> with open('file.json', 'rt') as f:
   ...     a = json.load(f, cls=ALPcouplingsDecoder)
