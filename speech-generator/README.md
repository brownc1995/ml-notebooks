# speech-generator

Here we use a RNN to generate speeches in the style of some input data.

The work found in this folder is heavily based on one of 
TensorFlow's [examples](https://www.tensorflow.org/tutorials/text/text_generation).
The training data can be found [here](https://github.com/ryanmcdermott/trump-speeches/blob/master/speeches.txt)
and is a transcript of each of Trump's 2016 campaign speeches. Trump is an obvious test-subject with his distinctive speech patterns.

## Example output

Below is an example output of the model (setting the parameter
`temperature=0.25`, all other parameters unadjusted):

> I have the best mind and they’re going to be back, I would have had a million – now, we have to build our infrastructure, our roads. We have to be smart enough to give a date – "We’re leaving on this date." And then they say "Why aren’t they paying us; what are we doing about that. <br/><br/>
So I’m very proud of me. But we’re going to be the smart people. We’re not going to let that can be so powerful that nobody is going to mess with us. Okay? Nobody ever talks about that.<br/><br/>
Now, Bush gave a very hard time frankly with what he said. But he said this before I have to say this so, so strongly, one of the press are too smart. We’re going to be a much bigger party, and I think we’re going to do this and that." I said, "Well, we have the best business leaders in this country because I’m telling you and this country was supposed to be there. You’ll see the report it. I think he’s a player here. I mean, there’s the worst. The worst. No, they’re very dishonest people.<br/><br/>
I have the best locations, because if they want...

## To Do
- add code for embedding projector in tensorboard

## References
- [Text generation with an RNN; TensorFlow](https://www.tensorflow.org/tutorials/text/text_generation)
