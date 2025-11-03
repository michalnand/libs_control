import numpy
import tracking_kalman_filter


class ObjectTracker:

    def __init__(self, max_objects, q_process_noise_var, r_measurement_noise_var, dt):
        self.max_objects = max_objects
        self.n_dims = 2

        self.kalman_filter = tracking_kalman_filter.TrackingKalmanFilter(q_process_noise_var, r_measurement_noise_var, self.n_dims, max_objects, dt)
        
        for n in range(self.max_objects):
            x_initial = numpy.random.randn(self.n_dims)
            self.kalman_filter.set_state(n, x_initial)

        self.cost_matrix = numpy.zeros((self.max_objects, self.max_objects))

    def step(self, x_obs):
        indices = []
        for n in range(self.n_dims):
            indices.append(n*3) 


        x_hat = self.kalman_filter.x_hat[:, indices]
        self._update_cost(x_obs, x_hat, alpha = 0.01)

        self.assignment = self._compute_assignemnt(self.cost_matrix)

        x_obs_assignemnt = x_obs[self.assignment]

        x_hat = self.kalman_filter.step(x_obs_assignemnt)

        return x_hat
    

    def prediction(self, n_steps):
        return self.kalman_filter.prediction(n_steps), self.assignment
    


    def _update_cost(self, x_obs, x_hat, alpha = 0.01):

        # distance matrix, rows : models, filter;  cols : objects
        d = ((numpy.expand_dims(x_obs, 0) - numpy.expand_dims(x_hat, 1))**2).sum(axis=-1)

        self.cost_matrix = (1.0 - alpha)*self.cost_matrix + alpha*d

    def _compute_assignemnt(self, cost_matrix):
        n_models, n_objects = cost_matrix.shape
        assignment  = numpy.full(n_models, -1, dtype=int)
        obj_taken   = numpy.zeros(n_objects, dtype=bool)

        # Precompute all (model, object, cost) and sort by cost
        models, objects = numpy.indices((n_models, n_objects))
        flat_pairs      = numpy.column_stack((models.ravel(), objects.ravel(), cost_matrix.ravel()))
        flat_pairs      = flat_pairs[numpy.argsort(flat_pairs[:, 2])]  # sort by cost

        assigned_count = 0
        idx = 0

        # Single while loop drives the process
        while assigned_count < n_objects and idx < len(flat_pairs):
            model, obj, cost = flat_pairs[idx]
            model, obj = int(model), int(obj)

            if assignment[model] == -1 and not obj_taken[obj]:
                assignment[model] = obj
                obj_taken[obj] = True
                assigned_count += 1

            idx += 1

        return assignment

   
