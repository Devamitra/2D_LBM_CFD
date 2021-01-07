import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class Lattice:
    def __init__(self, nx, ny, viscosity=1, temperature=298):
        self.nx = nx
        self.ny = ny
        self.bulk_velocity = np.zeros((nx+2, ny+2, 2), float)
        # self.bulk_velocity[1, :] = [1, 0]
        # self.bulk_velocity[self.ny, :] = [1, 0]
        # self.graph_u_streamline()
        # self.graph_u_vector()

        self.e = [np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.]), np.array([-1., 0.]), np.array([0., -1.]),
                  np.array([1., 1.]), np.array([-1., 1.]), np.array([-1., -1.]), np.array([1., -1.])]

        self.w = np.array([4/9., 1/9., 1/9., 1/9., 1/9., 1/36., 1/36., 1/36., 1/36.])

        self.f = np.zeros((nx+2, ny+2, 9), float)
        self.eq_f = np.zeros_like(self.f)

        self.rho = 1

        self.viscosity = viscosity
        self.relax_time = ((6 * self.viscosity) + 1) / 2

        self.obstacle = np.zeros((nx+2, ny+2), bool)

        self.equilibrium_function()
        self.initialise_f()
        self.new_velocity()

    def initialise_f(self):
        for x in range(self.nx+2):
            self.f[x, :] = self.eq_f[x, :].copy()

    def new_velocity(self):
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                temp = 0
                for i in range(9):
                    # print(self.f[x, y, i], self.e[i])
                    temp += self.f[x, y, i] * self.e[i]
                    # print(temp)
                self.bulk_velocity[x, y] = temp / self.rho

    def equilibrium_function(self):
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                u = self.bulk_velocity[x, y]
                s = np.zeros(9, float)
                for i in range(9):
                    e_dot_u = u.dot(self.e[i])
                    s[i] = (1 + (e_dot_u*3) + ((9/2.) * np.square(e_dot_u))
                            - ((3/2.) * u.dot(u)))

                self.eq_f[x, y] = self.rho * self.w * s

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

    def boundary_condition(self):
        self.bulk_velocity[1, :] = [1., 0]
        # self.bulk_velocity[self.nx+1, :] = [1., 0]

    def collision(self):
        for x in range(self.nx+2):
            for y in range(self.ny+2):
                self.f[x, y] = self.f[x, y] - ((self.f[x, y] - self.eq_f[x, y])/self.relax_time)

    def stream(self):
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

    def forcing_term(self):
        f_term = 8. * self.viscosity * 1 * self.rho / (6. * self.ny * self.ny)
        self.f[:, :, 1] += f_term
        self.f[:, :, 5] += f_term
        self.f[:, :, 8] += f_term

        self.f[:, :, 3] -= f_term
        self.f[:, :, 6] -= f_term
        self.f[:, :, 7] -= f_term

    def simulate(self, nt):
        for t in range(nt):
            self.stream()
            # self.refresh_values()
            self.new_velocity()
            self.boundary_condition()
            self.equilibrium_function()
            self.collision()
            # self.forcing_term()
            # self.graph_u_streamline()
            # self.graph_u_vector()

    def graph_rho(self):
        plt.imshow(self.rho, cmap="hot")
        plt.show()

    def graph_u_vector(self):
        x, y = np.meshgrid(np.linspace(1, self.nx+2, self.nx), np.linspace(1, self.ny+2, self.ny))
        Q = plt.quiver(y, x, self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 0], self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 1], scale=100, width=0.002)
        # plt.show()
        return Q, x, y

    def update_vector(self, num, Q, x, y):
        self.simulate(1)
        Q.set_UVC(self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 0], self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 1])
        return Q

    def graph_u_streamline(self):
        x, y = np.meshgrid(np.linspace(1, self.nx + 2, self.nx), np.linspace(1, self.ny + 2, self.ny))
        plt.streamplot(x, y, self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 0], self.bulk_velocity[1:self.nx + 1, 1:self.ny + 1, 1])
        plt.show()

    def set_f(self, old_f_val, tx, ty, i, fx, fy):
        if self.obstacle[tx, ty]:
            reflect = [0, 3, 4, 1, 2, 7, 8, 5, 6]
            self.f[fx, fy, reflect[i]] = old_f_val

        else:
            self.f[tx, ty, i] = old_f_val

    def set_obstacle(self, bool_matrix):
        self.obstacle = bool_matrix

    def set_tao(self, tao):
        self.relax_time = tao


if __name__ == "__main__":
    a = Lattice(20, 50, 10)
    # a.set_tao(100)
    matrix = np.zeros((22, 52), bool)
    # matrix[:, 0] = True
    # matrix[1, :] = True
    # matrix[20, :] = True
    # matrix[:, 21] = True
    matrix[20:30, :] = True

    a.set_obstacle(matrix)

    """
    # a.graph_u_streamline()
    a.graph_u_vector()
    a.simulate(10)
    # a.graph_u_streamline()
    plt.show()

    """

    fig = plt.figure(figsize=(15, 6))
    q, x, y = a.graph_u_vector()
    anim = animation.FuncAnimation(fig, a.update_vector, fargs=(q, x, y), interval=50, blit=False)
    plt.show()
    # """
