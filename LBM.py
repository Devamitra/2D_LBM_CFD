"""
module for simulation the motion an incompressible fluid using D2Q9 Lattice Boltzmann method
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class Lattice:
    """
    class creates lattice with both horizontal and vertical increment as 1

    Attributes
    -------------
    nx : int , number of column in the lattice
    ny : int , number of row in the lattice
    viscosity : float, viscosity of the fluid

    Methods
    ____________
    set_obstacle(bool_matrix) : takes a boolean matrix of the same size as lattice to represent an obstacle
    set_tao(tao) : sets the relaxation time externally

    """
    def __init__(self, nx, ny, viscosity=1/6., temperature=298):
        self.nx = nx  # number of cells, horizontal
        self.ny = ny  # number of cells, vertical

        # initialising the microscopic velocity, each coordinate holds an array representing vector[vertical,horizontal]
        self.bulk_velocity = np.zeros((nx+2, ny+2, 2), float)

        # self.bulk_velocity[self.ny, :] = [1, 0]
        # self.graph_u_streamline()
        # self.graph_u_vector()

        # defining the velocity for all 9 vectors in a 1x1 cell
        #    6   2    5
        #      \ |  /
        #  3 <-  0  -> 1
        #      / | \
        #    7   4   8
        self.e = [np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.]), np.array([-1., 0.]), np.array([0., -1.]),
                  np.array([1., 1.]), np.array([-1., 1.]), np.array([-1., -1.]), np.array([1., -1.])]

        # defining the weight of each vector, adds up to be 1
        self.w = np.array([4/9., 1/9., 1/9., 1/9., 1/9., 1/36., 1/36., 1/36., 1/36.])

        # defining the 3D array of floats holding distribution & equilibrium function of 9 vectors for each coordinates
        self.f = np.zeros((nx+2, ny+2, 9), float)  # distribution function
        self.eq_f = np.zeros_like(self.f)          # equilibrium function

        # defining the density, const since the fluid incompressible
        self.rho = 1

        # defining viscosity and relaxation time
        self.viscosity = viscosity
        self.relax_time = ((6 * self.viscosity) + 1) / 2

        # defining boolean matrix for an obstacle
        self.obstacle = np.zeros((nx+2, ny+2), bool)

        # initialising microscopic velocity so fluid flows from left to right
        self.bulk_velocity[1, :] = [1, 0]

        # initialising equilibrium function using microscopic velocity
        self.equilibrium_function()
        # initialising distribution using equilibrium function
        self.initialise_f()
        # working out new microscopic velocity using newly initialised distribution function
        self.new_velocity()

    def initialise_f(self):
        # initialises distribution function
        self.f = self.eq_f.copy()

    def new_velocity(self):
        # renews the microscopic velocity with current distribution function
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                temp = 0
                for i in range(9):
                    # print(self.f[x, y, i], self.e[i])
                    temp += self.f[x, y, i] * self.e[i]
                    # print(temp)
                # p = density , f = distribution function, x = position (vector), t = time
                # u = microscopic velocity (vector), e = velocity vectors in the cell (vector)
                # using p(x, t) = Sigma(f(x, t) of all 9 vectors) and
                # u(x, t) * p(x, t) = Sigma(f(x, t)* e(x, t) for all 9 vectors)
                self.bulk_velocity[x, y] = temp / self.rho

    def equilibrium_function(self):
        # renews the equilibrium function using current microscopic velocity
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                u = self.bulk_velocity[x, y]
                s = np.zeros(9, float)
                for i in range(9):
                    e_dot_u = u.dot(self.e[i])
                    s[i] = (1 + (e_dot_u*3) + ((9/2.) * np.square(e_dot_u))
                            - ((3/2.) * u.dot(u)))

                self.eq_f[x, y] = self.rho * self.w * s

    """
    def refresh_values(self):
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                temp = 0
                for i in range(9):
                    # print(self.f[x, y, i], self.e[i])
                    temp += self.f[x, y, i] * self.e[i]
                    # print(temp)
                self.bulk_velocity[x, y] = temp / self.rho

                # self.boundary_condition()

                u = self.bulk_velocity[x, y]
                s = np.zeros(9, float)
                for i in range(9):
                    e_dot_u = u.dot(self.e[i])
                    s[i] = (1 + (e_dot_u * 3) + ((9 / 2.) * np.square(e_dot_u))
                            - ((3 / 2.) * u.dot(u)))

                self.eq_f[x, y] = self.rho * self.w * s
                
    """

    def boundary_condition(self):
        # sets the preset boundary condition
        self.bulk_velocity[1, :] = [1., 0]
        # self.bulk_velocity[self.nx+1, :] = [1., 0]

    def collision(self):
        # collision step, renews all the distribution functions
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                # f = distribution function of a point, eq_f = equilibrium function of the same point
                # t = relaxation time, f* = new f
                # using f* = f - 1/t * (f - eq_f)
                # rearranged to get  f* = f*(1 - 1/t) + (1/t * eq_f)
                self.f[x, y] = (self.f[x, y]*(1-(1/self.relax_time))) + (self.eq_f[x, y] / self.relax_time)

    def stream(self):
        # reallocates the distribution function to new position, also deals with obstacle
        old_f = self.f.copy()
        # Horizontal
        self.f[:, :, 0] = old_f[:, :, 0]
        """
        for x in range(1, self.nx+1):
            self.f[x+1, :, 1] = old_f[x, :, 1]
            self.f[x-1, :, 3] = old_f[x, :, 3]
            self.f[x+1, 1:, 5] = old_f[x, :self.ny + 1, 5]
            self.f[x+1, :self.ny+1, 8] = old_f[x, 1:, 8]
            self.f[x-1, 1:, 6] = old_f[x, :self.ny + 1, 6]
            self.f[x-1, :self.ny+1, 7] = old_f[x, 1:, 7]

        for y in range(1, self.ny+1):
            self.f[:, y+1, 2] = old_f[:, y, 2]
            self.f[:, y-1, 4] = old_f[:, y, 4]

        del old_f


        """
        for x in range(1, self.nx+1):
            for y in range(1, self.ny+1):
                self.set_f(old_f[x, y, 1], x+1, y, 1, x, y)
                self.set_f(old_f[x, y, 2], x, y+1, 2, x, y)
                self.set_f(old_f[x, y, 3], x-1, y, 3, x, y)
                self.set_f(old_f[x, y, 4], x, y-1, 4, x, y)
                self.set_f(old_f[x, y, 5], x+1, y+1, 5, x, y)
                self.set_f(old_f[x, y, 6], x-1, y+1, 6, x, y)
                self.set_f(old_f[x, y, 7], x-1, y-1, 7, x, y)
                self.set_f(old_f[x, y, 8], x+1, y-1, 8, x, y)

        del old_f

    def set_f(self, old_f_val, tx, ty, i, fx, fy):
        if self.obstacle[tx, ty]:
            reflect = [0, 3, 4, 1, 2, 7, 8, 5, 6]
            self.f[fx, fy, reflect[i]] = old_f_val

        else:
            self.f[tx, ty, i] = old_f_val

    def forcing_term(self):
        # adds the forcing term to the distribution function
        # used if there is a force pushing the fluid horizontally
        f_term = 8. * self.viscosity * 1 * self.rho / (6. * self.ny * self.ny)
        self.f[:, :, 1] += f_term
        self.f[:, :, 5] += f_term
        self.f[:, :, 8] += f_term

        self.f[:, :, 3] -= f_term
        self.f[:, :, 6] -= f_term
        self.f[:, :, 7] -= f_term

    def simulate(self, nt):
        # runs the simulation for nt number of steps, where dt = 1
        for t in range(nt):
            self.stream()
            # self.refresh_values()
            self.new_velocity()
            # self.boundary_condition()
            self.equilibrium_function()
            self.collision()
            # self.forcing_term()
            # self.graph_u_streamline()
            # self.graph_u_vector()

    def graph_rho(self):
        # represents density as a heatmap
        plt.imshow(self.rho, cmap="hot")
        plt.show()

    def graph_u_vector(self, return_val=False):
        # graphs the microscopic velocity of the current time step
        # or initializes the vector field if return_val is True
        x, y = np.meshgrid(np.linspace(1, self.nx+2, self.nx), np.linspace(1, self.ny+2, self.ny))
        Q = plt.quiver(y, x, self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 0], self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 1], scale=100, width=0.002)
        if return_val:
            return Q, x, y

        plt.show()

    def update_vector(self, num, Q, x, y):
        # updates the vector in the vector field for animation
        self.simulate(1)
        Q.set_UVC(self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 0], self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 1])
        return Q

    def graph_u_streamline(self):
        # graphs the microscopic velocities as a streamline graph
        x, y = np.meshgrid(np.linspace(1, self.nx + 2, self.nx), np.linspace(1, self.ny + 2, self.ny))
        plt.streamplot(x, y, self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 0], self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 1])
        plt.show()

    def set_obstacle(self, bool_matrix):
        # sets the obstacle with boolean matrix
        self.obstacle = bool_matrix

    def set_tao(self, tao):
        # sets relaxation time and redefines viscosity accordingly, dt = 1
        self.relax_time = tao
        self.viscosity = ((2 * tao) - 1) / 6


if __name__ == "__main__":
    # help(__import__(__name__))

    a = Lattice(20, 20)
    a.set_tao(2.5)
    matrix = np.zeros((22, 22), bool)
    # matrix[:, 0] = True
    # matrix[1, :] = True
    # matrix[20, :] = True
    # matrix[:, 21] = True
    matrix[8:12, 8:12] = True

    a.set_obstacle(matrix)

    """
    # a.graph_u_streamline()
    a.graph_u_vector()
    a.simulate(10)
    # a.graph_u_streamline()
    plt.show()

    """

    fig = plt.figure(figsize=(15, 6))
    q, x, y = a.graph_u_vector(return_val=True)

    # runs the vector field as an animation
    anim = animation.FuncAnimation(fig, a.update_vector, fargs=(q, x, y), interval=50, blit=False)
    plt.show()
    # """
