import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers


class GravitationEnv(gym.Env):
    def __init__(self):
        super(GravitationEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )  # Actions represent forces in X, Y, and Z directions
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(9,), dtype=np.float32
        )  # Positions (X, Y, Z) and masses (m1, m2, m3)

        # Define initial state
        self.state = np.random.uniform(low=-5, high=5, size=(9,))

    def step(self, action):
        # Apply forces based on action
        force = action
        # Calculate gravitational force (simplified for one particle)
        net_force = np.array([0.0, 0.0, 0.0])
        for i in range(3):  # Assume 3 particles
            if i != self.target_particle:  # Ignore self-gravity
                r = (
                    self.state[:3] - self.state[i * 3 : i * 3 + 3]
                )  # Vector from other particle to target particle
                r_norm = np.linalg.norm(r)
                if r_norm > 1e-6:  # Avoid division by zero
                    f_gravity = (
                        self.state[i * 3 + 3] * self.state[3 + 3] / (r_norm**3)
                    ) * r
                    net_force += f_gravity

        # Update state (simplified Euler integration)
        self.state[:3] += self.state[3:] * 0.1  # Update positions based on velocities
        self.state[3:] += net_force * 0.1  # Update velocities based on forces

        # Calculate reward (simplified)
        reward = -np.linalg.norm(net_force)

        # Check if episode is done (simplified)
        done = False

        return self.state, reward, done, {}

    def reset(self):
        # Reset state to random initial conditions
        self.state = np.random.uniform(low=-5, high=5, size=(9,))
        return self.state

    def render(self, mode="human"):
        # Optional: Render environment
        pass


class ActorCritic(models.Model):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = self.build_actor(state_dim, action_dim)
        self.critic = self.build_critic(state_dim)

    def build_actor(self, state_dim, action_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(action_dim, activation="tanh")(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self, state_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model


class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.actor_optimizer = optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = optimizers.Adam(learning_rate=0.002)

    def get_action(self, state):
        return self.actor_critic.actor.predict(state[np.newaxis])[0]

    def train(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            next_actions = self.actor_critic.actor(next_states)
            next_q_values = self.actor_critic.critic([next_states, next_actions])
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
            q_values = self.actor_critic.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(q_values - target_q_values))

        critic_grads = tape.gradient(
            critic_loss, self.actor_critic.critic.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.actor_critic.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_actions = self.actor_critic.actor(states)
            actor_loss = -tf.reduce_mean(
                self.actor_critic.critic([states, new_actions])
            )

        actor_grads = tape.gradient(
            actor_loss, self.actor_critic.actor.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor_critic.actor.trainable_variables)
        )


# Training loop
env = GravitationEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = DDPG(state_dim, action_dim)

for _ in range(1000):  # Training episodes
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, reward, next_state, done)
        state = next_state
