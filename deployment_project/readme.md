### Bazel Build commands

```
1. Train and Save the Model:
  > bazel build //tensorflow_serving/example:sentiment_saved_model
  > bazel-bin/tensorflow_serving/example/sentiment_saved_model /tmp/sentiment_model

2. Start Server
  > bazel build //tensorflow_serving/model_servers:tensorflow_model_server
  > bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=sentiment --model_base_path=/tmp/sentiment_model/

3. Start Client
  > bazel build //tensorflow_serving/example:sentiment_client
  > bazel-bin/tensorflow_serving/example/sentiment_client --num_tests=1000 --server=localhost:9000

NOTE: For requesting server from remote machine(i.e not localhost) just replace localhost with the server's IP in the last command
```
### Reference

* [Common Bazel Build Installation Error](https://github.com/tensorflow/serving/issues/421)
