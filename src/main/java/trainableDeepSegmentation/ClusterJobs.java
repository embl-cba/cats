package trainableDeepSegmentation;

import com.jcraft.jsch.ChannelExec;
import com.jcraft.jsch.ChannelSftp;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.Session;
import java.io.*;

/**
 * Created by tischi on 26/07/17.
 */
public class ClusterJobs {


    public static void writeJobFile( String path )
    {
        try{
            PrintWriter writer = new PrintWriter(path, "UTF-8");
            writer.println("#!/bin/bash");
            writer.println("#SBATCH -N 1");
            writer.println("#SBATCH -n 1");
            writer.println("#SBATCH --mem 100M ");
            writer.println("#SBATCH -e slurm.%N.%j.err ");
            writer.println("#SBATCH -o slurm.%N.%j.out ");
            writer.println("touch /g/almf/tischer/yeah.job");
            writer.close();

            //module load X11
            //module load Java
            //xvfb-run -a /g/almfscreen/tischer/Fiji.app/ImageJ-linux64 -batch /g/almf/software/slurm_test_temp/hello.ijm

        } catch (IOException e) {
            System.out.println(e.toString());
        }
    }

    public static void checkJobs(String user, String password, String host, int port)
    {

        try
        {
            JSch jsch = new JSch();
            Session session = jsch.getSession(user, host, port);
            session.setPassword(password);
            session.setConfig("StrictHostKeyChecking", "no");
            System.out.println("Establishing Connection...");
            session.connect();
            System.out.println("Connection established.");

            ChannelExec channelExec = (ChannelExec) session.openChannel("exec");
            channelExec.setCommand("sacct -u " + user);

            InputStream in = channelExec.getInputStream();
            channelExec.setErrStream(System.err);
            channelExec.connect();

            String output = convertStreamToStr( in );
            System.out.println(output);

            channelExec.disconnect();
            session.disconnect();


        }
        catch(Exception e)
        {
            System.err.print(e);
        }

    }

    public static String convertStreamToStr(InputStream is) throws IOException {

        if (is != null) {
            Writer writer = new StringWriter();

            char[] buffer = new char[1024];
            try {
                Reader reader = new BufferedReader(new InputStreamReader(is,
                        "UTF-8"));
                int n;
                while ((n = reader.read(buffer)) != -1) {
                    writer.write(buffer, 0, n);
                }
            } finally {
                is.close();
            }
            return writer.toString();
        }
        else {
            return "";
        }
    }

    public static void main(String args[])
    {
        String user = "tischer";
        String password = "OlexOlex";
        String host = "login.cluster.embl.de";
        int port = 22;

        writeJobFile("/Volumes/almf/tischer/job-from-java.sh");

        try
        {
            String remoteFile="/g/almf/tischer/job-from-java.sh";
            JSch jsch = new JSch();
            Session session = jsch.getSession(user, host, port);
            session.setPassword(password);
            session.setConfig("StrictHostKeyChecking", "no");
            System.out.println("Establishing Connection...");
            session.connect();
            System.out.println("Connection established.");


            System.out.println("Crating SFTP Channel.");
            ChannelSftp channelSftp = (ChannelSftp) session.openChannel("sftp");
            channelSftp.connect();
            System.out.println("SFTP Channel created.");
            InputStream out = null;
            out = channelSftp.get(remoteFile);
            BufferedReader br = new BufferedReader(new InputStreamReader(out));
            String line;
            while ((line = br.readLine()) != null)
                System.out.println(line);
            br.close();
            channelSftp.disconnect();

            ChannelExec channelExec = (ChannelExec) session.openChannel("exec");
            channelExec.setCommand("sbatch /g/almf/tischer/job-from-java.sh");

            InputStream in = channelExec.getInputStream();
            channelExec.setErrStream(System.err);
            channelExec.connect();

            String output = convertStreamToStr( in );
            System.out.println( output );
            channelExec.disconnect();

            session.disconnect();

        }
        catch(Exception e)
        {
            System.err.print(e);
        }


        checkJobs(user, password, host, port);

    }
}
