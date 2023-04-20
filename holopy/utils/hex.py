import sys, numpy as np

def hexagon_diameter_to_coordinates( d, 
                                    x_spacing = 10.5/1000,
                                    y_spacing = 9/1000,
                                    f_tran = 0.02 
                            ) -> list( ( float, float, float ) ): 
    """
    Coordinate system for d-transducers diameter hexagon

    Centrepoint of central transducer is at origin (0,0,0)

    Array begins with the bottom left transducer.

    Args: 
        d:          diameter of hexagon (longest row) in transducer units 
        x_spacing:  interspacing between elements in the x axis
        y_spacing:  interspacing between elements in the y axis
        f_tran:     focal length of the PAT [m]
    """

    # from the diameter in transducer units (central and longest row) 
    # calculate array with transducers count 
    # for bottom row up to central row
    bottom_to_central_row_tran_count = np.arange( np.floor( (d+1)/2 ), np.floor( d+1 ), 1, dtype=int )

    # calculate array with rows' transducers count  
    rows_transducer_count = np.concatenate( ( bottom_to_central_row_tran_count, np.flip( bottom_to_central_row_tran_count )[1:] ), axis=0)

    # print(f" rows transducer count: { rows_transducer_count }")

    coords = []
    
    # for each row, 
    # depending on whether it is offset or not (i.e. shifted in relation to central row), 
    # calculate and assign X Y coordinates to each transducer
    for row, row_length in enumerate(rows_transducer_count):

        for elem in range( row_length ):
            
            coord_x = x_spacing * ( elem - row_length/2 + .5 )

            coord_y = -sys.maxsize-1

            if d % 2 != 0:
                coord_y = y_spacing * ( row - (d-1)/2 )
            else:
                coord_y = y_spacing * ( row - d/2 )
                
            coords.append( (coord_x, coord_y, f_tran) )  
    
    return coords



def pesb_hex( evpd, resolution, coords ) -> tuple:

    tx = np.array([coord[0] for coord in coords], dtype=float)
    ty = np.array([coord[1] for coord in coords], dtype=float)
    tz = np.array([coord[2] for coord in coords], dtype=float)

    # building evaluation plane points
    ev = np.linspace( -evpd/2, evpd/2, resolution ) # create vector with desired resolution
    ex, ey = np.meshgrid(ev, ev)

    # x, y & z vectors for evaluation-plane sample points:
    px, py = ex.flatten(), ey.flatten()
    pz = np.zeros(len(ex)*len(ey))

    # Grids to describe the vector distances between each transducer & evaluation plane sample point.
    txv, pxv = np.meshgrid(tx, px)
    tyv, pyv = np.meshgrid(ty, py)
    tzv, pzv = np.meshgrid(tz, pz)

    rxyz = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2 + (tzv-pzv)**2) # Pythagoras for xyz distances
    rxy = np.sqrt((txv-pxv)**2 + (tyv-pyv)**2) # Pythagoras for xy distances
    
    return rxyz, rxy