host = "172.16.0.171"
port = 22
username = "aditya.singh"
password = "Welcome#123"
import paramiko
from paramiko.transport import SecurityOptions, Transport


class SFTPClient:
    '''
    Class to connect with the remote server using SFTP server
    '''
    _connection = None

    def __init__(self, host, port, username, password):
        """
        Initialize the sftp connection
        Args:
            host (str): ip of the sftp server
            port (int): port of the server
            username (str): username of server
            password (str): password of server
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.root_path = "/models"

        self.OpenConnection(self.host, self.port, self.username, self.password)

    @classmethod
    def OpenConnection(self, host, port, username, password):
        '''
        Open Connection with remote server using SFTp protocol
        Args:
            host (str): ip of the sftp server
            port (int): port of the server
            username (str): username of server
            password (str): password of server
        '''
        transport = Transport(sock=(host, port))
        transport.connect(username=username, password=password)
        self._connection = paramiko.SFTPClient.from_transport(transport)

    def download(self, remote_path, local_path):
        '''
        Download the data from remote server to local 
        Args:
            remote_path (str): path of the model on the server
            local_path (str): local path to download the model
        '''
        print("===>", remote_path)
        print("local path===>", local_path)
        self._connection.get(remote_path, local_path, callback=None)

    def downloadpytorch(self, filenm, destpath):
        '''
        Download pytorch model
        Args:
            filenm: filename of the model
            destpath: local path
        '''
        srcpath = self.root_path + "/pytorch/" + filenm
        destpath = destpath + filenm
        self.download(srcpath, destpath)

    def downloadtf(self, filenm, destpath):
        '''
        Download Tensorflow model
        Args:
            filenm: filename of the model
            destpath: local path
        '''
        srcpath = self.root_path + "/tensorflow/" + filenm
        destpath = destpath
        self.download(srcpath, destpath)

    def downloadyolov5(self, filenm, destpath):
        '''
        Downaload Yolov5 Model
        Args:
            filenm: filename of the model
            destpath: local path
        '''
        srcpath = self.root_path + "/yolov5/" + filenm
        destpath = destpath
        self.download(srcpath, destpath)

    def downloadyolov8(self, srcpath, destpath):
        '''
        Download Yolov8 model
        Args:
            srcpath: source path of model
            destpath: local path of model
        '''
        srcpath = self.root_path + "/yolov8/" + filenm
        destpath = destpath + filenm
        self.download(srcpath, destpath)


# sf=SFTPClient(host, port, username, password)
# sf.downloadtf("Person.zip","model/tg.zip")
