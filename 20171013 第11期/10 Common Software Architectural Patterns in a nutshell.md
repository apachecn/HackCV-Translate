# 10 Common Software Architectural Patterns in a nutshell

原文链接：[10 Common Software Architectural Patterns in a nutshell](https://towardsdatascience.com/10-common-software-architectural-patterns-in-a-nutshell-a0b47a1e9013?from=hackcv&hmsr=hackcv.com)

Ever wondered how large enterprise scale systems are designed? Before major software development starts, we have to choose a suitable architecture that will provide us with the desired functionality and quality attributes. Hence, we should understand different architectures, before applying them to our design.



![img](https://cdn-images-1.medium.com/max/800/1*M22DR3WPqbWXWidYIq2GwA.png)

### What is an Architectural Pattern?

According to Wikipedia,

> An **architectural pattern** is a general, reusable solution to a commonly occurring problem in software architecture within a given context. Architectural patterns are similar to software design pattern but have a broader scope.

In this article, I will be briefly explaining the following 10 common architectural patterns with their usage, pros and cons.

1. **Layered pattern**
2. **Client-server pattern**
3. **Master-slave pattern**
4. **Pipe-filter pattern**
5. **Broker pattern**
6. **Peer-to-peer pattern**
7. **Event-bus pattern**
8. **Model-view-controller pattern**
9. **Blackboard pattern**
10. **Interpreter pattern**

### 1. Layered pattern

This pattern can be used to structure programs that can be decomposed into groups of subtasks, each of which is at a particular level of abstraction. Each layer provides services to the next higher layer.

The most commonly found 4 layers of a general information system are as follows.

- **Presentation layer** (also known as **UI layer**)
- **Application layer** (also known as **service layer**)
- **Business logic layer** (also known as **domain layer**)
- **Data access layer** (also known as **persistence layer**)

#### Usage

- General desktop applications.
- E commerce web applications.



![img](https://cdn-images-1.medium.com/max/800/1*jMWk_JqqyyloVPhTs_Zd1A.png)

Layered pattern

### 2. Client-server pattern

This pattern consists of two parties; a **server** and multiple **clients**. The server component will provide services to multiple client components. Clients request services from the server and the server provides relevant services to those clients. Furthermore, the server continues to listen to client requests.

#### Usage

- Online applications such as email, document sharing and banking.



![img](https://cdn-images-1.medium.com/max/800/1*4xX_WQQuD2u0PMK5bcWFkQ.png)

Client-server pattern

### 3. Master-slave pattern

This pattern consists of two parties; **master** and **slaves**. The master component distributes the work among identical slave components, and computes a final result from the results which the slaves return.

#### Usage

- In database replication, the master database is regarded as the authoritative source, and the slave databases are synchronized to it.
- Peripherals connected to a bus in a computer system (master and slave drives).



![img](https://cdn-images-1.medium.com/max/800/1*lsK9QntZl2d5oLojwRGXDg.png)

Master-slave pattern

### 4. Pipe-filter pattern

This pattern can be used to structure systems which produce and process a stream of data. Each processing step is enclosed within a **filter** component. Data to be processed is passed through **pipes**. These pipes can be used for buffering or for synchronization purposes.

#### Usage

- Compilers. The consecutive filters perform lexical analysis, parsing, semantic analysis, and code generation.
- Workflows in bioinformatics.



![img](https://cdn-images-1.medium.com/max/800/1*qikehZcDhhl_wWsqeI_nvg.png)

Pipe-filter pattern

### 5. Broker pattern

This pattern is used to structure distributed systems with decoupled components. These components can interact with each other by remote service invocations. A **broker** component is responsible for the coordination of communication among **components**.

Servers publish their capabilities (services and characteristics) to a broker. Clients request a service from the broker, and the broker then redirects the client to a suitable service from its registry.

#### Usage

- Message broker software such as [**Apache ActiveMQ**](https://en.wikipedia.org/wiki/Apache_ActiveMQ), [**Apache Kafka**](https://en.wikipedia.org/wiki/Apache_Kafka), [**RabbitMQ**](https://en.wikipedia.org/wiki/RabbitMQ) and [**JBoss Messaging**](https://en.wikipedia.org/wiki/JBoss_Messaging).



![img](https://cdn-images-1.medium.com/max/800/1*1qRQZjLRAd0yY_T9p2OgBw.png)

Broker pattern

### 6. Peer-to-peer pattern

In this pattern, individual components are known as **peers**. Peers may function both as a **client**, requesting services from other peers, and as a **server**, providing services to other peers. A peer may act as a client or as a server or as both, and it can change its role dynamically with time.

#### Usage

- File-sharing networks such as [**Gnutella**](https://en.wikipedia.org/wiki/Gnutella) and [**G2**](https://en.wikipedia.org/wiki/Gnutella2))
- Multimedia protocols such as [**P2PTV**](https://en.wikipedia.org/wiki/P2PTV) and [**PDTP**](https://en.wikipedia.org/wiki/Peer_Distributed_Transfer_Protocol).



![img](https://cdn-images-1.medium.com/max/800/1*ROvkckSTw1UncrbQSmUJUQ.png)

Peer-to-peer pattern

### 7. Event-bus pattern

This pattern primarily deals with events and has 4 major components; **event source**, **event listener**, **channel** and **event bus**. Sources publish messages to particular channels on an event bus. Listeners subscribe to particular channels. Listeners are notified of messages that are published to a channel to which they have subscribed before.

#### Usage

- Android development
- Notification services



![img](https://cdn-images-1.medium.com/max/800/1*DOZ4nVR9zkJm-EnXT3KOGQ.png)

Event-bus pattern

### 8. Model-view-controller pattern

This pattern, also known as MVC pattern, divides an interactive application in to 3 parts as,

1. **model** — contains the core functionality and data
2. **view** — displays the information to the user (more than one view may be defined)
3. **controller** — handles the input from the user

This is done to separate internal representations of information from the ways information is presented to, and accepted from, the user. It decouples components and allows efficient code reuse.

#### Usage

- Architecture for World Wide Web applications in major programming languages.
- Web frameworks such as [**Django** ](https://en.wikipedia.org/wiki/Django_%28web_framework%29)and [**Rails**](https://en.wikipedia.org/wiki/Ruby_on_Rails).



![img](https://cdn-images-1.medium.com/max/800/1*OP0CS6O5Sb66jpc-H-IuRQ.png)

Model-view-controller pattern

### 9. Blackboard pattern

This pattern is useful for problems for which no deterministic solution strategies are known. The blackboard pattern consists of 3 main components.

- **blackboard** — a structured global memory containing objects from the solution space
- **knowledge source** — specialized modules with their own representation
- **control component** — selects, configures and executes modules.

All the components have access to the blackboard. Components may produce new data objects that are added to the blackboard. Components look for particular kinds of data on the blackboard, and may find these by pattern matching with the existing knowledge source.

#### Usage

- Speech recognition
- Vehicle identification and tracking
- Protein structure identification
- Sonar signals interpretation.



![img](https://cdn-images-1.medium.com/max/800/1*ArbMx7A21I47llvwUTiSDg.png)

Blackboard pattern

### **10. Interpreter pattern**

This pattern is used for designing a component that interprets programs written in a dedicated language. It mainly specifies how to evaluate lines of programs, known as sentences or expressions written in a particular language. The basic idea is to have a class for each symbol of the language.

#### Usage

- Database query languages such as SQL.
- Languages used to describe communication protocols.



![img](https://cdn-images-1.medium.com/max/800/1*DrC3T5R4SsdcQY6aXLCRZA.png)

Interpreter pattern

### Comparison of Architectural Patterns

The table given below summarizes the pros and cons of each architectural pattern.



![img](https://cdn-images-1.medium.com/max/1000/1*Z9dKeyf6yi0nFMaUZF1P3Q.png)

Comparison of Architectural Patterns

Hope you found this article useful. I would love to hear your thoughts. 😇

Thanks for reading. 😊

Cheers! 😃