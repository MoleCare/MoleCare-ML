https://www.confluent.io/blog/build-deploy-scalable-machine-learning-production-apache-kafka/

https://www.confluent.io/blog/using-apache-kafka-drive-cutting-edge-machine-learning/
On a high level, a machine learning lifecycle contains of two different parts:
- Model training: In this step, we feed historical data into an algorithm to learn patterns from the past. The result is an analytic model.
- Generating predictions: In this step, we use an analytic model for making predictions on new events based on the learned pattern.
Machine learning is a continuous process, where we repeatedly improve and redeploy the analytic model over time.

https://devcenter.heroku.com/articles/kafka-on-heroku
- A Kafka cluster is comprised of a number of brokers, or instances running Kafka. The number of brokers in a cluster can be scaled to increase capacity, resilience, and parallelism.
- Brokers manage streams of messages (events sent to Kafka) in topics. Topics are configured with a range of options (retention or compaction, replication factor, etc) dependent on the data they are meant to support.
- Topics are comprised of a number of partitions, discrete subsets of a topic used to balance the concerns of parallelism and ordering. Increased numbers of partitions can increase the number of producers and consumers that can work on a given topic, increasing parallelism and throughput. Messages within a partition are ordered, but the ordering of messages across partitions is not guaranteed. Balancing needs of parallelism and ordering is key to proper partition configuration for a topic.



