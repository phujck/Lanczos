import numpy as np
import sys

class Lattice:

    def __init__(self,nx,ny,lat_type,nnn=False,bc=None):

        if isinstance(bc,str):
            assert(bc.lower() != 'obc') # obc not currently working
        self.nx = nx
        self.ny = ny
        if ny == 0:
            self.dim = 1
        else:
            self.dim = 2
        self.lat_type = lat_type

        # This will set up and store 
        # self.nsites
        # self.bc
        # self.max_nn             The max number of nearest neighbours
        # self.unit_vectors       The two unit vectors which define the lattice
        # self.lattice_pos        (x,y) Position of lattice site
        # self.neighbours         (0-)index of nearest-neighbour lattice sites (list of lists)
        # self.neighbours_dir     A list of dictionaries giving the index of the lattice site and boundary phase for a given direction (contains all the information of neighbours, but resolved for direction)
        # self.boundary_phase     +- phase of nearest-neighbour due to boundary conditions
        self.setup_lattice(bc,nnn)
        
        self.ordered_dists = None   # These are not set up in init
        self.ordered_inds = None

        if nnn:
            # Setup next-nearest-neighbour lists
            
            # self.max_nnn          The max number of nnn from a site
            # self.nnneighbours     (0-)index of next-nearest-neighbour lattice sites
            # self.nnn_phase        +- phase of next-nearest-neighbour due to boundary conditions
            self.setup_nnn()
        else:
            self.nnneighbours = None
            self.nnn_phase = None

    def setup_lattice(self,bc=None,nnn=False):

        if self.lat_type.lower() == 'single_site':
            self.setup_singlesite_lat()
        elif self.ny == 0:
            # 1D lattice
            self.setup_1DHub(bc,nnn)
        elif self.lat_type.lower() == 'square':
            self.setup_2DHub_square()
        elif self.lat_type.lower() == 'tilted':
            self.setup_2DHub_tilted(bc,nnn)
        elif self.lat_type.lower() == 'dimer':
            self.setup_dimerhub()
        else:
            print('Unrecognised lattice')
            assert(0)

        return

    def setup_singlesite_lat(self):
        ''' A "ficticious" single site lattice so that it can be tiled if necessary'''

        print('Setting up "ficticious" single site lattice')
        self.nsites = 1
        self.nx = 1
        self.ny = 1
        self.bc = None
        self.max_nn = 0
        self.lattice_pos = [(0, 0)]
        self.unit_vectors = np.asarray([[1,0],[0,1]])
        self.neighbours = [[]] 
        self.boundary_phase = [[]] 
        self.neighbours_dir = [dict()]
        return

    def find_disp_ind(self,ind,disp):
        ''' Find the index of the site on the lattice a displacement 
        vector (disp) from the current site indexed "ind" '''

        if isinstance(disp,int):
            disp_ = (disp,0)
            assert(self.dim == 1)
        else:
            disp_ = disp

        new_site = ind 
        new_phase = 1
        # Find the index in the lattice of the site (x,y) from plaq_origin
        for nx in range(abs(disp_[0])):
            if disp_[0] > 0:
                new_site = self.neighbours_dir[new_site][(1,0)][0]
                new_phase *= self.neighbours_dir[new_site][(1,0)][1]
            else:
                new_site = self.neighbours_dir[new_site][(-1,0)][0]
                new_phase *= self.neighbours_dir[new_site][(-1,0)][1]
        for ny in range(abs(disp_[1])):
            if disp_[1] > 0:
                new_site = self.neighbours_dir[new_site][(0,1)][0]
                new_phase *= self.neighbours_dir[new_site][(0,1)][1]
            else:
                new_site = self.neighbours_dir[new_site][(0,-1)][0]
                new_phase *= self.neighbours_dir[new_site][(0,-1)][1]
        return new_site, new_phase

    def setup_nnn(self):

        print('Setting up next-nearest-neighbour lists')
        self.nnneighbours = [[] for x in range(self.nsites)]
        self.nnn_phase = [[] for x in range(self.nsites)]
        self.nnneighbours_dir = [{} for x in range(self.nsites)]

        if self.dim == 1:
            assert(self.nsites >= 4)
            self.max_nnn = 2
            vectors = [(2,0),(-2,0)]
            for i in range(self.nsites):
                site_px = self.neighbours_dir[self.neighbours_dir[i][(1,0)][0]][(1,0)][0]
                self.nnneighbours[i].append(site_px)
                self.nnn_phase[i].append(self.neighbours_dir[i][(1,0)][1] * self.neighbours_dir[self.neighbours_dir[i][(1,0)][0]][(1,0)][1])
                self.nnneighbours_dir[i][(2,0)] = (site_px, self.nnn_phase[i][-1])
                site_mx = self.neighbours_dir[self.neighbours_dir[i][(-1,0)][0]][(-1,0)][0]
                self.nnneighbours[i].append(site_mx)
                self.nnn_phase[i].append(self.neighbours_dir[i][(-1,0)][1] * self.neighbours_dir[self.neighbours_dir[i][(-1,0)][0]][(-1,0)][1])
                self.nnneighbours_dir[i][(-2,0)] = (site_mx, self.nnn_phase[i][-1])
        else:
            assert(self.nsites > 2)
            if(self.lat_type.lower() == 'tilted' and self.nx == self.ny == 2 and self.bc.lower() == 'abc'):
                print('Cannot have tilted 2x2 system with nnn hopping and APBCs, since then you have sites connected to the same site through different boundaries (and hence signs)')
                assert(0)
            self.max_nnn = 4
            vectors = [(1,1),(-1,1),(1,-1),(-1,-1)]
            for i in range(self.nsites):
                # +x, +y
                self.nnneighbours[i].append(self.neighbours_dir[self.neighbours_dir[i][(1,0)][0]][(0,1)][0])
                self.nnn_phase[i].append(self.neighbours_dir[i][(1,0)][1] * self.neighbours_dir[self.neighbours_dir[i][(1,0)][0]][(0,1)][1])
                self.nnneighbours_dir[i][(1,1)] = (self.nnneighbours[i][-1], self.nnn_phase[i][-1])
                # +x, -y
                self.nnneighbours[i].append(self.neighbours_dir[self.neighbours_dir[i][(1,0)][0]][(0,-1)][0])
                self.nnn_phase[i].append(self.neighbours_dir[i][(1,0)][1] * self.neighbours_dir[self.neighbours_dir[i][(1,0)][0]][(0,-1)][1])
                self.nnneighbours_dir[i][(1,-1)] = (self.nnneighbours[i][-1], self.nnn_phase[i][-1])
                # -x, +y
                self.nnneighbours[i].append(self.neighbours_dir[self.neighbours_dir[i][(-1,0)][0]][(0,1)][0])
                self.nnn_phase[i].append(self.neighbours_dir[i][(-1,0)][1] * self.neighbours_dir[self.neighbours_dir[i][(-1,0)][0]][(0,1)][1])
                self.nnneighbours_dir[i][(-1,1)] = (self.nnneighbours[i][-1], self.nnn_phase[i][-1])
                # -x, -y
                self.nnneighbours[i].append(self.neighbours_dir[self.neighbours_dir[i][(-1,0)][0]][(0,-1)][0])
                self.nnn_phase[i].append(self.neighbours_dir[i][(-1,0)][1] * self.neighbours_dir[self.neighbours_dir[i][(-1,0)][0]][(0,-1)][1])
                self.nnneighbours_dir[i][(-1,-1)] = (self.nnneighbours[i][-1], self.nnn_phase[i][-1])
        
        
        # Remove duplicates in neighbour list
        # Note - they are kept in the dictionary
        for i in range(self.nsites):
            if (len(self.nnneighbours[i]) != len(set(self.nnneighbours[i]))):
                # Duplicate indices in list
                newlist = []
                newlist_bound = []
                for ind1,nei in enumerate(self.nnneighbours[i]):
                    # Only add if hasn't been added before
                    if nei not in self.nnneighbours[i][:ind1]:
                        newlist.append(nei)
                        newlist_bound.append(self.nnn_phase[i][ind1])
                self.nnneighbours[i] = newlist
                self.nnn_phase[i] = newlist_bound
        
        print('Next-nearest-neighbour index list of sites: ')
        write_neighbour_list(self.nsites,vectors,self.nnneighbours_dir)
        return

    def setup_dimerhub(self):
        print('Setting up 1D hubbard dimer with {} sites per monomer'.format(self.nx//2))
        assert(self.nx % 2 == 0)
        self.nsites = self.nx
        self.max_nn = 2
        self.lattice_pos = np.asarray([[x,y] for x,y in zip(range(self.nsites),[0]*self.nsites)])
        self.unit_vectors = None

    def setup_1DHub(self,bc=None,nnn=False):
        print('Setting up real-space 1D lattice with regular cell')
        self.nsites = self.nx
        print('Total number of sites in lattice: ',self.nsites)
        if bc is None:
            # Choose boundary conditions automatically to get CS determinant
            if ((self.nx //2) % 2 == 0):
                print('Choosing anti-periodic boundary conditions')
                self.bc = 'abc'
            else:
                print('Choosing periodic boundary conditions')
                self.bc = 'pbc'
            if nnn and self.nx <= 4:
                print('Lattice too small for anti-periodic boundary conditions with nnn hopping. Reverting to PBCs')
                self.bc = 'pbc'
        elif bc.lower() == 'open':
            print('Open boundary conditions specified')
            self.bc = 'open'
        elif bc.lower() == 'pbc':
            print('Periodic boundary conditions specified')
            self.bc = 'pbc'
        elif bc.lower() == 'abc':
            print('Anti-periodic boundary conditions specified')
            self.bc = 'abc'
        else:
            sys.exit(1)
        self.lattice_pos = np.asarray([[x,y] for x,y in zip(range(self.nsites),[0]*self.nsites)])
        self.unit_vectors = np.asarray([[1,0],[0,0]])
        self.max_nn = 2
        self.neighbours = [[] for x in range(self.nsites)]
        self.boundary_phase = [[] for x in range(self.nsites)]
        self.neighbours_dir = [dict() for x in range(self.nsites)]
        for i in range(1,self.nsites-1):
            self.neighbours[i] = [i+1, i-1]
            self.neighbours_dir[i][(1,0)] = (i+1,1)
            self.neighbours_dir[i][(-1,0)] = (i-1,1)
            self.boundary_phase[i] = [1, 1]
        self.neighbours[0] = [1, self.nsites-1]
        self.neighbours[self.nsites-1] = [0, self.nsites-2]
        self.neighbours_dir[0][(1,0)] = (1,1)
        self.neighbours_dir[self.nsites-1][(-1,0)] = (self.nsites-2,1)
        if self.bc == 'pbc':
            self.boundary_phase[0] = [1, 1]
            self.boundary_phase[self.nsites-1] = [1, 1]
            self.neighbours_dir[0][(-1,0)] = (self.nsites-1,1) 
            self.neighbours_dir[self.nsites-1][(1,0)] = (0, 1)
        elif self.bc == 'abc':
            self.boundary_phase[0] = [1, -1]
            self.boundary_phase[self.nsites-1] = [-1, 1]
            self.neighbours_dir[0][(-1,0)] = (self.nsites-1,-1) 
            self.neighbours_dir[self.nsites-1][(1,0)] = (0, -1)
        else:
            self.boundary_phase[0] = [1, 0]
            self.boundary_phase[self.nsites-1] = [0, 1]

        # Remove duplicates in neighbour list
        # Note - they are kept in the dictionary
        for i in range(self.nsites):
            if (len(self.neighbours[i]) != len(set(self.neighbours[i]))):
                # Duplicate indices in list
                newlist = []
                newlist_bound = []
                for ind1,nei in enumerate(self.neighbours[i]):
                    # Only add if hasn't been added before
                    if nei not in self.neighbours[i][:ind1]:
                        newlist.append(nei)
                        newlist_bound.append(self.boundary_phase[i][ind1])
                self.neighbours[i] = newlist
                self.boundary_phase[i] = newlist_bound
        
        for i in range(self.nsites):
            for a in self.neighbours[i]:
                if i not in self.neighbours[a]:
                    print('Error in neighbours list')
                    print('Lattice sites i a {} {}'.format(i,a))
                    print('Neighbours for site i: {}'.format(self.neighbours[i]))
                    print('Neighbours for site a: {}'.format(self.neighbours[a]))
                    assert(0)
        
        # All coordinates in the positive quadrant
        print('Lattice sites: ')
        print('Lattice index : lattice coord')
        for i in range(len(self.lattice_pos)):
            print(i,self.lattice_pos[i])

        print('Neighbour index list of sites: ')
        write_neighbour_list(self.nsites,[(1,0),(-1,0)],self.neighbours_dir)
        return

    def setup_2DHub_square(self):
        
        print('Setting up real-space 2D square lattice with regular cell')
        self.nsites = self.nx*self.ny
        print('Total number of sites in lattice: ',self.nsites)
        print('Choosing periodic boundary conditions')
        self.bc = 'pbc'
        bc = self.bc
        nx = self.nx
        ny = self.ny
        self.max_nn = 4

        self.lattice_pos = []
        x = []
        y = []
        for l1 in range(nx):
            for l2 in range(ny):
                self.lattice_pos.append((l1,l2))
                x.append(l1)
                y.append(l2)
        self.unit_vectors = np.asarray([[1,0],[0,1]])

        if len(self.lattice_pos) != self.nsites:
            sys.exit('Error generating lattice sites')

        # All coordinates in the positive quadrant
        print('Lattice sites: ')
        for i in range(len(self.lattice_pos)):
            print(i,self.lattice_pos[i][0],self.lattice_pos[i][1])
        
        # find neighbours for each lattice
        self.neighbours = [[] for i in range(self.nsites)]
        self.boundary_phase = [[] for i in range(self.nsites)]
        self.neighbours_dir = [dict() for i in range(self.nsites)]
        # boundary conditions
        # nbc=1 normal neighbours, i.e. no sign change etc. (one for each step
        # i.e. mx,px...)
        # bc=1 no sign change, normal neighbours
        # bc=-1 sign change for antiperiodic boundary conditions
        # bc = 0 for open boundary conditions, i.e. no neighbours
        mxbc = 1
        mybc = 1
        pxbc = 1
        pybc = 1
        b1 = nx
        b2 = ny
        for (i,(x1,y1)) in enumerate(self.lattice_pos):
            px = x1 + 1
            mx = x1 - 1
            py = y1 + 1
            my = y1 - 1
            # boundary conditions
            if px > np.amax(x):
                if bc == 'pbc':
                    px = np.amin(x)
                    pxbc = 1
                elif bc == 'abc':
                    # sign change
                    px = np.amin(x)
                    pxbc = -1
                elif bc == 'obc':
                    # no neighbours
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    pxbc = 0
            if mx < np.amin(x):
                if bc == 'pbc':
                    mx = np.amax(x)
                    mxbc = 1
                elif bc == 'abc':
                    # sign change
                    mx = np.amax(x)
                    mxbc = -1
                elif bc == 'obc':
                    # no neighbours
                    #self.neighbours[i,1] = -1
                    #self.boundary_phase[i,1] = 0
                    mxbc = 0
            if py > np.amax(y):
                if bc == 'pbc':
                    py = np.amin(y)
                    pybc = 1
                elif bc == 'abc':
                    # sign change
                    py = np.amin(y)
                    pybc = -1
                elif bc == 'obc':
                    # no neighbours
                    #self.neighbours[i,2] = -1
                    #self.boundary_phase[i,2] = 0
                    pybc = 0
            if my < np.amin(y):
                if bc == 'pbc':
                    my = np.amax(y)
                    mybc = 1
                elif bc == 'abc':
                    # sign change
                    my = np.amax(y)
                    mybc = -1
                elif bc == 'obc':
                    # no neighbours
                    #self.neighbours[i,3] = -1
                    #self.boundary_phase[i,3] = 0
                    mybc = 0
            for (k,(x2,y2)) in enumerate(self.lattice_pos):
                # x-coordinate: up,down
                if x2 == px and y2 == y1:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(pxbc) 
                    self.neighbours_dir[i][(1,0)] = (k,pxbc)
                if x2 == mx and y2 == y1:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(mxbc)
                    self.neighbours_dir[i][(-1,0)] = (k,mxbc)
                # y-coordinate: up,down
                if x2 == x1 and y2 == py:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(pybc)
                    self.neighbours_dir[i][(0,1)] = (k,pybc)
                if x2 == x1 and y2 == my:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(mybc)
                    self.neighbours_dir[i][(0,-1)] = (k,mybc)
            # for normal neighbours
            pxbc = 1
            mxbc = 1
            pybc = 1
            mybc = 1

        # Remove duplicates in neighbour list
        # Note - they are kept in the dictionary
        #print 'Removing duplicates: '
        for i in range(self.nsites):
            #print 'site: ',i,self.neighbours[i]
            if (len(self.neighbours[i]) != len(set(self.neighbours[i]))):
                #print 'Duplicate found'
                # Duplicate indices in list
                newlist = []
                newlist_bound = []
                for ind1,nei in enumerate(self.neighbours[i]):
                    # Only add if hasn't been added before
                    if nei not in self.neighbours[i][:ind1]:
                        newlist.append(nei)
                        newlist_bound.append(self.boundary_phase[i][ind1])
                self.neighbours[i] = newlist
                self.boundary_phase[i] = newlist_bound
                #print 'Replaced by: ',self.neighbours[i]

        # if one lattice i site occurs in the neighbour list lattice site a
        # then lattice site a should also be in the neighbour list 
        # of lattice site i
        for i in range(self.nsites):
            for a in self.neighbours[i]:
                if i not in self.neighbours[a]:
                    print('Error in neighbours list')
                    print('Lattice sites i a {} {}'.format(i,a))
                    print('Neighbours for site i: {}'.format(self.neighbours[i]))
                    print('Neighbours for site a: {}'.format(self.neighbours[a]))
                    assert(0)
        
        print('Neighbour index list of sites: ')
        write_neighbour_list(self.nsites,[(1,0),(-1,0),(0,1),(0,-1)],self.neighbours_dir)
#        print 'Neighbour index list of sites: '
#        print 'index, neighbours (+x, -x, +y, -y): '
#        for i in range(self.nsites):
#            if self.boundary_phase[i,0] == -1:
#                neigh_plusx = ' -' + str(self.neighbours[i,0])
#            elif self.boundary_phase[i,0] == 1:
#                neigh_plusx = ' +' + str(self.neighbours[i,0])
#            else:
#                neigh_plusx = ' *** ' 
#            if self.boundary_phase[i,1] == -1:
#                neigh_minx = ' -' + str(self.neighbours[i,1])
#            elif self.boundary_phase[i,1] == 1:
#                neigh_minx = ' +' + str(self.neighbours[i,1])
#            else:
#                neigh_minx = ' *** ' 
#            if self.boundary_phase[i,2] == -1:
#                neigh_plusy = ' -' + str(self.neighbours[i,2])
#            elif self.boundary_phase[i,2] == 1:
#                neigh_plusy = ' +' + str(self.neighbours[i,2])
#            else:
#                neigh_plusy = ' *** ' 
#            if self.boundary_phase[i,3] == -1:
#                neigh_miny = ' -' + str(self.neighbours[i,3])
#            elif self.boundary_phase[i,3] == 1:
#                neigh_miny = ' +' + str(self.neighbours[i,3])
#            else:
#                neigh_miny = ' *** ' 
#            print i,neigh_plusx,neigh_minx,neigh_plusy,neigh_miny

        return

    def setup_2DHub_tilted(self,bc=None,nnn=False):

        print('Setting up real-space 2D square lattice with 45deg tilted cell')
        print('Note that a unit cell has two sites')
        # Note that for a 45 degree tilted square lattice, the number of lattice
        # sites is L = 2n^2
        self.nsites = 2*self.nx*self.ny
        print('Total number of sites in lattice: ',self.nsites)
        if bc is None: 
            if self.nx % 2 == 1:
                print('Choosing periodic boundary conditions')
                self.bc = 'pbc'
                bc = 'pbc'
            else:
                print('Choosing anti-periodic boundary conditions')
                self.bc = 'abc'
                bc = 'abc'
                if nnn and (self.nx == 2 or self.ny == 2):
                    print('Warning: This cell may be too small for APBCs with nnn hopping, since the resulting 1e ham may not be hermitian?')
        else:
            if bc.lower() == 'pbc':
                print('Setting periodic boundary conditions')
                self.bc = 'pbc'
            else:
                print('Choosing anti-periodic boundary conditions')
                self.bc = 'abc'
        nx = self.nx
        ny = self.ny
        self.max_nn = 4
        
        self.lattice_pos = []
        for l1 in range(-nx,nx+1):
            #for l2 in range(2*(b2+1)):
            for l2 in range(-ny,ny+1):
                if (l1 <= 0 and l2 < 0 and l2 >= -l1-ny) or (l1 >= 0 and l2 < 0 and l2 >= l1-ny) or (l1 <=0 and l2 >= 0 and l2 < l1+ny) or (l1 >= 0 and l2 >=0 and l2 < -l1+ny):
                    self.lattice_pos.append((l1,l2))

        if len(self.lattice_pos) != self.nsites:
            sys.exit('Error generating lattice sites')
        self.unit_vectors = np.asarray([[1,1],[1,-1]])

        print('Lattice sites: ')
        for i in range(len(self.lattice_pos)):
            print(i,self.lattice_pos[i][0],self.lattice_pos[i][1])
        
        # find neighbours for each lattice
        self.neighbours = [[] for x in range(self.nsites)]
        self.boundary_phase = [[] for x in range(self.nsites)]
        self.neighbours_dir = [dict() for x in range(self.nsites)]
        # boundary conditions
        # nbc=1 normal neighbours, i.e. no sign change etc. (one for each step
        # i.e. mx,px...)
        # bc=1 no sign change, normal neighbours
        # bc=-1 sign change for antiperiodic boundary conditions
        # bc = 0 for open boundary conditions, i.e. no neighbours
        mxbc = 1
        mybc = 1
        pxbc = 1
        pybc = 1
        b1 = nx
        b2 = ny
        for (i,(x1,y1)) in enumerate(self.lattice_pos):
            px = []
            mx = []
            py = []
            my = []
            px.append(x1 + 1)
            px.append(y1)
            mx.append(x1 - 1)
            mx.append(y1)
            py.append(x1)
            py.append(y1 + 1)
            my.append(x1)
            my.append(y1 - 1)
            # boundary conditions
            if px[1] < 0 and px[0] > px[1]+b2:
                if bc == 'pbc':
                    px[1] = px[1] + b2 
                    px[0] = px[1] - b2 + 1
                    pxbc = 1
                elif bc == 'abc':
                    # sign change
                    px[1] = px[1] + b2
                    px[0] = px[1] - b2 + 1
                    pxbc = -1
                elif bc == 'obc':
                    # no neighbours
                    px[0] = 0
                    px[1] = 0
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    pxbc = 0
            if px[1] >= 0 and px[0] >= -px[1]+b2:
                if bc == 'pbc':
                    px[1] = px[1] - b2
                    px[0] = -px[1] - b2
                    pxbc = 1
                elif bc == 'abc':
                    # sign change
                    px[1] = px[1] - b2
                    px[0] = -px[1] - b2
                    pxbc = -1
                elif bc == 'obc':
                    # no neighbours
                    px[0] = 0
                    px[1] = 0
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    pxbc = 0
            if mx[1] < 0 and mx[0] < -mx[1]-b2:
                if bc == 'pbc':
                    mx[1] = mx[1] + b2 
                    mx[0] = -mx[1] + b2 - 1
                    mxbc = 1
                elif bc == 'abc':
                    # sign change
                    mx[1] = mx[1] + b2
                    mx[0] = -mx[1] + b2 - 1
                    mxbc = -1
                elif bc == 'obc':
                    # no neighbours
                    mx[0] = 0
                    mx[1] = 0
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    mxbc = 0
            if mx[1] >= 0 and mx[0] <= mx[1]-b2:
                if bc == 'pbc':
                    mx[1] = mx[1] - b2
                    mx[0] = mx[1] + b2
                    mxbc = 1
                elif bc == 'abc':
                    # sign change
                    mx[1] = mx[1] - b2
                    mx[0] = mx[1] + b2
                    mxbc = -1
                elif bc == 'obc':
                    # no neighbours
                    mx[0] = 0
                    mx[1] = 0
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    mxbc = 0
            if py[0] < 0 and py[1] >= py[0]+b2:
                if bc == 'pbc':
                    py[0] = py[0] + b2
                    py[1] = py[0] - b2
                    pybc = 1
                elif bc == 'abc':
                    # sign change
                    py[0] = py[0] + b2
                    py[1] = py[0] - b2
                    pybc = -1
                elif bc == 'obc':
                    # no neighbours
                    py[0] = 0
                    py[1] = 0
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    pybc = 0
            if py[0] >= 0 and py[1] >= -py[0]+b2:
                if bc == 'pbc':
                    if py[0] != 0:
                        py[0] = py[0] - b2
                    py[1] = -py[0] - b2
                    pybc = 1
                elif bc == 'abc':
                    # sign change
                    if py[0] != 0:
                        py[0] = py[0] - b2
                    py[1] = -py[0] - b2
                    pybc = -1
                elif bc == 'obc':
                    # no neighbours
                    py[0] = 0
                    py[1] = 0
                    #self.neighbours[i,0] = -1
                    #self.boundary_phase[i,0] = 0
                    pybc = 0
            if my[0] >= 0 and my[1] < my[0]-b2:
                if bc == 'pbc':
                    if my[0] != 0:
                        my[0] = my[0] - b2
                    my[1] = my[0] + b2 - 1
                    mybc = 1
                elif bc == 'abc':
                    # sign change
                    if my[0] != 0:
                        my[0] = my[0] - b2
                    my[1] = my[0] + b2 - 1
                    mybc = -1
                elif bc == 'obc':
                    # no neighbours
                    my[0] = 0 
                    my[1] = 0
                    #self.neighbours[i,1] = -1
                    #self.boundary_phase[i,1] = 0
                    mybc = 0
            if my[0] < 0 and my[1] < -x1-b2:
                if bc == 'pbc':
                    my[0] = my[0] + b2
                    my[1] = -my[0] + b2 - 1
                    mybc = 1
                elif bc == 'abc':
                    # sign change
                    my[0] = my[0] + b2
                    my[1] = -my[0] + b2 - 1
                    mybc = -1
                elif bc == 'obc':
                    # no neighbours
                    my[0] = 0
                    my[1] = 0
                    #self.neighbours[i,1] = -1
                    #self.boundary_phase[i,1] = 0
                    mybc = 0
 
            for k,(x,y) in enumerate(self.lattice_pos):
                # x-coordinate: up,down
                if x == px[0] and y == px[1]:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(pxbc)
                    self.neighbours_dir[i][(1,0)] = (k,pxbc)
                if x == mx[0] and y == mx[1]:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(mxbc)
                    self.neighbours_dir[i][(-1,0)] = (k,mxbc)
                # y-coordinate: up,down
                if x == py[0] and y == py[1]:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(pybc)
                    self.neighbours_dir[i][(0,1)] = (k,pybc)
                if x == my[0] and y == my[1]:
                    self.neighbours[i].append(k)
                    self.boundary_phase[i].append(mybc)
                    self.neighbours_dir[i][(0,-1)] = (k,mybc)
            # for normal neighbours
            pxbc = 1
            mxbc = 1
            pybc = 1
            mybc = 1

        # Remove duplicates in neighbour list
        # Note - they are kept in the dictionary
        for i in range(self.nsites):
            if (len(self.neighbours[i]) != len(set(self.neighbours[i]))):
                # Duplicate indices in list
                newlist = []
                newlist_bound = []
                for ind1,nei in enumerate(self.neighbours[i]):
                    # Only add if hasn't been added before
                    if nei not in self.neighbours[i][:ind1]:
                        newlist.append(nei)
                        newlist_bound.append(self.boundary_phase[i][ind1])
                self.neighbours[i] = newlist
                self.boundary_phase[i] = newlist_bound

        # if one lattice i site occurs in the neighbour list lattice site a
        # then lattice site a should also be in the neighbour list 
        # of lattice site i
        for i in range(self.nsites):
            for a in self.neighbours[i]:
                if i not in self.neighbours[a]:
                    print('Error in neighbours list')
                    print('Lattice sites i a {} {}'.format(i,a))
                    print('Neighbours for site i: {}'.format(self.neighbours[i]))
                    print('Neighbours for site a: {}'.format(self.neighbours[a]))
                    assert(0)
        
        print('Neighbour index list of sites: ')
        write_neighbour_list(self.nsites,[(1,0),(-1,0),(0,1),(0,-1)],self.neighbours_dir)

        return

    def create_site_dists(self):
        ''' Creates a list of site indices from each site, ordered by increasing distance, and with an accompanying 
            minimum image translation vector between the points, as well as a separate list with the distances'''

        self.ordered_inds = np.zeros((3,self.nsites,self.nsites),dtype=np.int_,order='F')
        self.ordered_dists = np.zeros((self.nsites,self.nsites),dtype=np.float_)

        for i in range(self.nsites):
            i_vec = np.asarray(self.lattice_pos[i])
            for j in range(self.nsites):
                #print 'considering root site: ',i,' to ',j

                # How far is site j from site i?
                j_vec = np.asarray(self.lattice_pos[j])
                jvecs = []
                jvecs.append(j_vec)
                # Also consider all single lattice translations of j and find the minimal image
                jvecs.append(j_vec + self.nx*self.unit_vectors[0])
                jvecs.append(j_vec - self.nx*self.unit_vectors[0])
                jvecs.append(j_vec + self.ny*self.unit_vectors[1])
                jvecs.append(j_vec - self.ny*self.unit_vectors[1])
                jvecs.append(j_vec + self.nx*self.unit_vectors[0] + self.ny*self.unit_vectors[1])
                jvecs.append(j_vec - self.nx*self.unit_vectors[0] + self.ny*self.unit_vectors[1])
                jvecs.append(j_vec + self.nx*self.unit_vectors[0] - self.ny*self.unit_vectors[1])
                jvecs.append(j_vec - self.nx*self.unit_vectors[0] - self.ny*self.unit_vectors[1])

                dists = []
                for k,k_vec in enumerate(jvecs):
                    dists.append(np.sqrt(np.sum((i_vec-k_vec)**2)))
                    #print ' image: ',(k_vec-i_vec),np.sqrt(np.sum((k_vec-i_vec)**2))
                dist_ind = np.argmin(np.asarray(dists))

                self.ordered_inds[0,j,i] = j
                # Minimium image vector
                self.ordered_inds[1:,j,i] = jvecs[dist_ind] - i_vec
                self.ordered_dists[i,j] = dists[dist_ind]

            #print 'Before ordering, from site {}, the nearest sites are'.format(i)
            #for j in range(self.nsites):
            #    print self.ordered_inds[0,j,i],tuple(self.ordered_inds[1:,j,i])

            # Now order the sites by distance
            idx = self.ordered_dists[i,:].argsort()
            self.ordered_inds[:,:,i] = self.ordered_inds[:,idx,i]
            self.ordered_dists[i,:] = self.ordered_dists[i,idx]
            print('for site ',i,' ordered indices, vectors, dists are: ')
            for j in range(self.nsites):
                print(self.ordered_inds[0,j,i],tuple(self.ordered_inds[1:,j,i]),self.ordered_dists[i,j])

        return

    def setup_neighbourlist_fort(self):
        ''' Setup a contiguous array of neighbours of each lattice site for easy interfacing to fortran. 
            Creates: self.neighbourlst_fort[4,nsites] where the 4 elements refer to the index of the elements
                                                      in the +x, -x, +y, -y directions in the lattice.'''

        self.neighbourlst_fort = np.zeros((self.nsites,4),dtype=np.int_,order='F')
        for i in range(self.nsites):
            if (1,0) in self.neighbours_dir[i]:
                self.neighbourlst_fort[i,0] = self.neighbours_dir[i][(1,0)][0]
            else:
                self.neighbourlst_fort[i,0] = i # Do not go anywhere...
            if (-1,0) in self.neighbours_dir[i]:
                self.neighbourlst_fort[i,1] = self.neighbours_dir[i][(-1,0)][0]
            else:
                self.neighbourlst_fort[i,1] = i # Do not go anywhere...
            if (0,1) in self.neighbours_dir[i]:
                self.neighbourlst_fort[i,2] = self.neighbours_dir[i][(0,1)][0]
            else:
                self.neighbourlst_fort[i,2] = i # Do not go anywhere...
            if (0,-1) in self.neighbours_dir[i]:
                self.neighbourlst_fort[i,3] = self.neighbours_dir[i][(0,-1)][0]
            else:
                self.neighbourlst_fort[i,3] = i # Do not go anywhere...

        return

def create_2e_ham(lat,tHub,U_val = 0.0, J_val = 0.0):

    n = lat.nsites
    if tHub:
        # Two-electron hamiltonian
        eri = np.zeros((n,n,n,n))
        for i in range(n):
            eri[i,i,i,i] = U_val
        return eri
    else:
        # Ficticious Heisenberg model modelled as a Fermionic system
        eri_aa = np.zeros((n,n,n,n))
        eri_ab = np.zeros((n,n,n,n))
        eri_bb = np.zeros((n,n,n,n))

        for i in range(n):
            for nei in lat.neighbours[i]:
                # (spin)-density-density interaction (Anti-Ferromagnetic interaction)
                eri_aa[i,i,nei,nei] = -J_val/4.0
                eri_bb[i,i,nei,nei] = -J_val/4.0
                eri_ab[i,i,nei,nei] = J_val/4.0
                # spin-flip terms
                eri_ab[i,nei,nei,i] = -J_val/4.0
        return (eri_aa, eri_ab, eri_bb)

def create_1e_ham(lat,tHub,t_prime = 0.0):
    
    n = lat.nsites
    if abs(t_prime) > 1.0e-10:
        tinc_nnn = True
    else:
        tinc_nnn = False

    if tHub:
        h1 = np.zeros((n,n))

        for i in range(n):
            for j,nei in enumerate(lat.neighbours[i]):
                h1[i,nei] = -1.0 * lat.boundary_phase[i][j]

            if tinc_nnn:
                for k,nnnei in enumerate(lat.nnneighbours[i]):
                    #print 'setting nnn h element: ',i,nnnei,lat.nnn_phase[i][k],t_prime * lat.nnn_phase[i][k]
                    h1[i,nnnei] = t_prime * lat.nnn_phase[i][k]
        assert(np.allclose(h1-h1.T,np.zeros_like(h1)))
        return h1
    else:
        # Heisenberg model. Since it must be UHF, return two h1 arrays
        h1_a = np.zeros((n,n))
        h1_b = np.zeros((n,n))
        return (h1_a, h1_b)

def tile_ham_largerlat(lat_small,lat_large,h_large,h_small,logger=False):

    if lat_small.lat_type is not 'single_site':
        assert(lat_large.lat_type == lat_small.lat_type)
    assert(lat_large.nx % lat_small.nx == 0)
    if not (lat_large.ny == 0 and lat_small.ny == 0):
        assert(lat_large.ny % lat_small.ny == 0)

    h_tiled = h_large.copy()

    if lat_small.lat_type == 'single_site':
        assert(lat_large.nx == lat_large.ny)
        nx_repeats = ny_repeats = lat_large.nsites // 2
    else:
        nx_repeats = lat_large.nx // lat_small.nx
        if lat_small.ny == 0:
            ny_repeats = 1 
        else:
            ny_repeats = lat_large.ny // lat_small.ny
    if logger:
        print('nx repeats: ',nx_repeats)
        print('ny repeats: ',ny_repeats)

    zerosite_origin = np.asarray(lat_large.lattice_pos[0])-np.asarray(lat_small.lattice_pos[0])
    if logger: print('Vector from origin of large lattice to origin of small lattice: ',zerosite_origin)

    plaquette_vectors = []
    if lat_small.lat_type == 'single_site':
        for i in range(len(lat_large.lattice_pos)):
            plaquette_vectors.append(np.asarray(lat_large.lattice_pos[i])-zerosite_origin)
    else:
        for i in range(nx_repeats):
            for j in range(ny_repeats):
                plaquette_vectors.append(lat_small.nx*i*lat_large.unit_vectors[0]+lat_small.ny*j*lat_large.unit_vectors[1])

    if logger: 
        print('Vectors in large lattice to origin of all repeats of small lattice:')
        for i in range(len(plaquette_vectors)):
            print(i,plaquette_vectors[i])

    smalllat_vectors = []
    for i in range(lat_small.nsites):
        # These are the (x,y) movements to take you from the 0-site of the small
        # lattice to the other lattice sites in this plaquette
        smalllat_vectors.append(np.asarray(lat_small.lattice_pos[i])-np.asarray(lat_small.lattice_pos[0]))

    if logger: 
        print('Vectors within the small lattice from the origin to each site: ')
        for i in range(len(smalllat_vectors)):
            print(i,smalllat_vectors[i])

    ind_tiled = [False]*lat_large.nsites

    for (vec,plaq_disp) in enumerate(plaquette_vectors):
        # Set to be the '0'-site in this plaquette on the large lattice
        plaq_origin = lat_large.find_disp_ind(0,plaq_disp)[0]
        if logger: print('Plaquette index origin: ',plaq_origin)
        for (i,smalllat_disp) in enumerate(smalllat_vectors):
            # Find the index in the large lattice of the site (x,y) from plaq_origin
            ind_large_lat = lat_large.find_disp_ind(plaq_origin,smalllat_disp)[0]
            assert(not ind_tiled[ind_large_lat])
            ind_tiled[ind_large_lat] = True
            for j in range(i): 
                ind_large_lat_2 = lat_large.find_disp_ind(plaq_origin,smalllat_vectors[j])[0]
                assert(i != j)
                assert(ind_large_lat != ind_large_lat_2)
                #print 'i,j: ',i,j, ind_large_lat,ind_large_lat_2

                h_tiled[ind_large_lat,ind_large_lat_2] += h_small[i,j]
                h_tiled[ind_large_lat_2,ind_large_lat] += h_small[j,i]

            h_tiled[ind_large_lat,ind_large_lat] += h_small[i,i]

    assert(np.all(ind_tiled))
    #print h_tiled
    #assert(np.allclose(h_tiled-h_tiled.T,np.zeros_like(h_tiled)))
    return h_tiled

def write_neighbour_list(nsites,vectors,neighbour_dir):
    ''' Write out the list of vectors from each site, and the index of the site they point to. 
    In: nsites
        vectors:    A list of vector tuples giving the directions in the lattice
        neighbour_dir:  The list of dictionaries of neighbours'''

    assert(nsites == len(neighbour_dir))
    print('Index, vector, neighbour (sign is phase)')
    for i in range(nsites):
        string = ''
        for j,vec in enumerate(vectors):
            if vec in neighbour_dir[i]:
                if neighbour_dir[i][vec][1] == -1:
                    string += str(vec) + ': -' + str(neighbour_dir[i][vec][0]) + '  '
                else:
                    string += str(vec) + ': +' + str(neighbour_dir[i][vec][0]) + '  '
            else:
                string += str(vec) + ': None  '
        print(i,string)
    return
